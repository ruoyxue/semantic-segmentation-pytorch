import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .utils import create_scale_mask
from .modules import Mlp

class Local_Scale_Attention(nn.Module):
    def __init__(self, 
                 window_sizes, 
                 dim=32, 
                 num_heads=8, 
                 scale_attn_drop=0.0, 
                 scale_proj_drop=0.0,
                 qkv_bias=True, 
                 qk_scale=None,
                 use_rsb=True, 
                 attn_mask=None,
                 vis=False):
        super(Local_Scale_Attention, self).__init__()

        # params
        self.dim = dim
        self.num_heads = num_heads # num of attention head
        self.sizes = window_sizes
        self.square_sizes = [self.sizes[i] ** 2  for i in range(len(self.sizes))]
        self.use_rsb = use_rsb
        self.mask = attn_mask
        self.vis = vis
        self.scale = qk_scale or sum(self.square_sizes) ** -0.5

        # modules
        self.scale_qkv = nn.Linear(self.dim, self.dim * 3, bias = qkv_bias)
        self.scale_softmax = nn.Softmax(dim=-1)
        self.scale_attn_drop = nn.Dropout(scale_attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(scale_proj_drop)

        # build scale relative positional bias
        self.relative_scale_bias_table = nn.Parameter(torch.zeros(2 * (sum(self.square_sizes) + sum(self.square_sizes[1:])) - 1, num_heads))
        # get pair-wise relative position index for each token inside the scale space
        coords_h_scale = torch.arange(sum(self.square_sizes))
        coords_w_scale = torch.arange(sum(self.square_sizes)).flip(0)
        offset = [0]
        for i in range(len(self.square_sizes) - 1):
            offset.append(sum(self.square_sizes[-i - 1:]))
        offset_h_scale = []
        offset_w_scale = []
        for i in range(len(self.square_sizes)):
            offset_w_scale.extend([offset[- i - 1] for _ in range(self.square_sizes[i])])
            offset_h_scale.extend([offset[i] for _ in range(self.square_sizes[i])])
        offset_h_scale = torch.tensor(offset_h_scale).unsqueeze(-1).repeat(1, sum(self.square_sizes))
        offset_w_scale = torch.tensor(offset_w_scale).unsqueeze(0).repeat(sum(self.square_sizes), 1)
        coords_scale = torch.stack(torch.meshgrid([coords_h_scale, coords_w_scale]))  # 2, len_code, len_code
        relative_scale_index = coords_scale.sum(0) + offset_h_scale + offset_w_scale  # len_code, len_code
        self.register_buffer("relative_scale_index", relative_scale_index)
        trunc_normal_(self.relative_scale_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        assert N == sum(self.square_sizes) and C == self.dim
        qkv = self.scale_qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # build relative scale bias
        if self.use_rsb:
            relative_scale_bias = self.relative_scale_bias_table[self.relative_scale_index.view(-1)].view(
                                                                sum(self.square_sizes), sum(self.square_sizes), -1)  # Wh*Ww,Wh*Ww,nH
            relative_scale_bias = relative_scale_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            attn = attn + relative_scale_bias.unsqueeze(0)
        
        # build scale mask
        if self.mask is not None:
            scale_mask = create_scale_mask([self.square_sizes[i] for i in range(len(self.square_sizes))], [1 for _ in range(len(self.square_sizes))], 
                                           -100, self.mask).to(x.device)
            scale_mask.requires_grad = False
            attn = attn + scale_mask
        
        attn = self.scale_softmax(attn) # num_pyramid*B, num_heads, layers_per_head, layers_per_head
        attn = self.scale_attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.vis:
            return x, attn
        else:
            return x

class Local_Scale_Unification_Block(nn.Module):
    def __init__(self, 
                 window_sizes, 
                 dim=32, 
                 num_heads=8, 
                 mlp_ratio=4.,
                 scale_attn_drop=0.0, 
                 scale_proj_drop=0.0,
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 use_rsb=True, 
                 attn_mask=None,
                 vis=False):
        super(Local_Scale_Unification_Block, self).__init__()

        # params
        self.dim = dim
        self.num_heads = num_heads # num of attention head
        self.sizes = window_sizes
        self.square_sizes = [self.sizes[i] ** 2  for i in range(len(self.sizes))]
        self.use_rsb = use_rsb
        self.mask = attn_mask
        self.vis = vis

        # modules
        self.norm1 = norm_layer(dim)
        self.attn = Local_Scale_Attention(
                 window_sizes, 
                 dim=self.dim, 
                 num_heads=self.num_heads, 
                 scale_attn_drop=scale_attn_drop, 
                 scale_proj_drop=scale_proj_drop,
                 qkv_bias=qkv_bias, 
                 qk_scale=qk_scale,
                 use_rsb=self.use_rsb, 
                 attn_mask=self.mask,
                 vis=self.vis)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=scale_proj_drop, use_lspe=False)
    
    def forward(self, x):
        if self.vis:
            # x = self.attn(self.norm1(x))[0]
            x = x + self.drop_path(self.attn(self.norm1(x))[0])
            attn = self.attn(self.norm1(x))[1]
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            # x = self.attn(self.norm1(x))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.autograd.Variable(torch.randn(4489, 85, 32)).to(device)
    lsa = Local_Scale_Unification_Block([8, 4, 2, 1], attn_mask='Block').to(device)
    f = lsa(input)

    num_params = 0
    for param in lsa.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
