from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .utils import create_spatial_mask
from .modules import DWConv, Mlp, Scale_AvgPooling, Scale_UnPooling, Scale_MaxPooling, MultiScale_DWConv

class Global_Scale_Attention(nn.Module):
    def __init__(self, 
                 window_sizes, 
                 dim=32, 
                 num_heads=8, 
                 spatial_attn_drop=0.0, 
                 spatial_proj_drop=0.0,
                 qkv_bias=True, 
                 qk_scale=None,
                 use_lspe=True,
                 attn_mask=None,
                 lspe_k_size=[3, 5, 7],
                 vis=False):
        super(Global_Scale_Attention, self).__init__()
        # params
        self.dim = dim
        self.num_heads = num_heads # num of attention head
        self.sizes = window_sizes
        self.square_sizes = [self.sizes[i] ** 2  for i in range(len(self.sizes))]
        self.mask = attn_mask
        self.scale = qk_scale
        self.use_lspe = use_lspe
        self.lspe_k_size = lspe_k_size
        self.vis = vis

        # modules
        # self.scale_avgpooling = Scale_AvgPooling(self.sizes)
        # self.scale_unpooling = Scale_UnPooling(self.sizes)
        self.spatial_qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.spatial_softmax = nn.Softmax(dim=-1)
        self.spatial_attn_drop = nn.Dropout(spatial_attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(spatial_proj_drop)

        # self.relative_spatial_bias_table = nn.Parameter(torch.zeros((2 * 32 - 1) * (2 * 32 - 1), num_heads))

        if self.use_lspe:
            self.lspe_channel_squeezing = nn.Linear(sum(self.square_sizes) * self.dim // self.num_heads, 1, bias=True)
            self.lspe_dwconv = nn.Conv2d(self.num_heads, self.num_heads * 2, self.lspe_k_size, 1, self.lspe_k_size // 2, bias=True, groups=self.num_heads)
            # self.lspe_dwconvs = MultiScale_DWConv(self.lspe_k_sizes, self.num_heads, self.num_heads * 2)
            self.lspe_avgpooling = nn.AdaptiveAvgPool2d(1)
            self.lspe_maxpooling = nn.AdaptiveMaxPool2d(1)
            self.lspe_ln = nn.LayerNorm(self.num_heads)
    
    def forward(self, x, batch_size, H=None, W=None):
        B_, _, C = x.shape
        assert B_ % batch_size == 0 and C == self.dim
        qkv = self.spatial_qkv(x).view(batch_size, B_ // batch_size, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5).flatten(-2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.scale or (B_ // batch_size) ** -0.5
        q = q * scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_lspe:
            assert H is not None and W is not None and H * W == B_ // batch_size
            x_2d = x.detach().clone().view(batch_size, H, W, -1, self.num_heads, C // self.num_heads).permute(0, 4, 1, 2, 3, 5).flatten(-2)
            x_2d = x.view(batch_size, H, W, -1, self.num_heads, C // self.num_heads).permute(0, 4, 1, 2, 3, 5).flatten(-2)
            x_2d = self.lspe_channel_squeezing(x_2d).squeeze(-1)

            x_2d = (self.lspe_dwconv(x_2d) + self.lspe_avgpooling(x_2d).repeat(1, 2, H, W) + 
                    self.lspe_maxpooling(x_2d).repeat(1, 2, H, W)).view(batch_size, 2, self.num_heads, H, W).transpose(0, 1)
            
            # x_weight, x_proj = x_2d[0].mean(0).flatten(-2).unsqueeze(-1), x_2d[1].mean(0).flatten(-2).unsqueeze(-2)
            # relative_spatial_bias = (x_weight @ x_proj).permute(1, 2, 0)
            # relative_spatial_bias = self.lspe_ln(relative_spatial_bias).permute(2, 0, 1).contiguous()

            x_weight, x_proj = x_2d[0].mean(0).unsqueeze(1), x_2d[1].mean(0).unsqueeze(0)
            relative_spatial_bias = F.conv_transpose2d(x_proj, x_weight, groups=self.num_heads)
            relative_spatial_bias = relative_spatial_bias.squeeze(0).flatten(-2).transpose(0, 1)

            # # normalize
            # ones_weight, ones_proj = torch.ones((1, 1, H, W)), torch.ones((1, 1, H, W))
            # count = F.conv_transpose2d(ones_proj, ones_weight).to(x.device)
            # count.requires_grad = False
            # relative_spatial_bias = (relative_spatial_bias / count).squeeze(0).flatten(-2).transpose(0, 1)

            coords_h_spatial = torch.arange(H)
            coords_w_spatial = torch.arange(W)
            coords_spatial = torch.stack(torch.meshgrid([coords_h_spatial, coords_w_spatial]))  # 2, Wh, Ww
            coords_spatial_flatten = torch.flatten(coords_spatial, 1)  # 2, Wh*Ww
            relative_coords_spatial = coords_spatial_flatten[:, :, None] - coords_spatial_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords_spatial = relative_coords_spatial.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords_spatial[:, :, 0] += H - 1  # shift to start from 0
            relative_coords_spatial[:, :, 1] += W - 1
            relative_coords_spatial[:, :, 0] *= 2 * W - 1
            relative_spatial_index = relative_coords_spatial.sum(-1)  # Wh*Ww, Wh*Ww

            relative_spatial_bias = relative_spatial_bias[relative_spatial_index.view(-1)].view(H * W, H * W, -1).permute(2, 0, 1).contiguous()
            # relative_spatial_bias = self.lspe_ln(relative_spatial_bias).permute(2, 0, 1).contiguous()

            attn = attn + relative_spatial_bias.unsqueeze(0)

            # ak = self.dwconv(k_flat).flatten(2).view(batch_size, self.num_heads, -1, B_ // batch_size).transpose(2, 3)
            # k += ak
            # pos = torch.zeros(((2 * H - 1) * (2 * W - 1), self.num_heads)).to(x.device)
            # for i in range(self.num_heads):
            #     pos[0 : H * W, i] += x_r[]

        if self.mask is not None:
            spatial_mask = create_spatial_mask(B_ // batch_size, -100).to(x.device)
            spatial_mask.requires_grad = False
            attn = attn + spatial_mask

        attn = self.spatial_softmax(attn) # num_pyramid*B, num_heads, layers_per_head, layers_per_head
        attn = self.spatial_attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().view(B_, self.num_heads, -1, C // self.num_heads).permute(0, 2, 1, 3).flatten(-2)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.vis:
            return x, attn
        else:
            return x

class Global_Scale_Unification_Block(nn.Module):
    def __init__(self, 
                 window_sizes, 
                 dim=32, 
                 num_heads=8, 
                 mlp_ratio=4.,
                 spatial_attn_drop=0.0, 
                 spatial_proj_drop=0.0,
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 use_lspe=True, 
                 lspe_k_size=7,
                 attn_mask=None,
                 vis=False):
        super(Global_Scale_Unification_Block, self).__init__()

        # params
        self.dim = dim
        self.num_heads = num_heads # num of attention head
        self.sizes = window_sizes
        self.square_sizes = [self.sizes[i] ** 2  for i in range(len(self.sizes))]
        self.use_lspe = use_lspe
        self.mask = attn_mask
        self.vis = vis

        # modules
        self.norm1 = norm_layer(dim)
        self.attn = Global_Scale_Attention(
                 window_sizes, 
                 dim=self.dim, 
                 num_heads=self.num_heads, 
                 spatial_attn_drop=spatial_attn_drop, 
                 spatial_proj_drop=spatial_proj_drop,
                 qkv_bias=qkv_bias, 
                 qk_scale=qk_scale,
                 use_lspe=self.use_lspe,
                 attn_mask=self.mask,
                 lspe_k_size=lspe_k_size,
                 vis=self.vis)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=spatial_proj_drop, use_lspe=self.use_lspe, lspe_k_size=lspe_k_size)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=spatial_proj_drop, use_lspe=False)
        # if self.use_lspe:
        #     self.peg = DWConv(k_size=lspe_k_size, dim=dim)
            # self.linear = nn.Sequential(
            #     nn.Linear(dim, dim, bias=True),
            #     act_layer()
            # )
    
    def forward(self, x, batch_size):
        B_, _, C = x.shape
        assert B_ % batch_size == 0 and C == self.dim
        H = W = int(math.sqrt(B_ // batch_size))
        if self.vis:
            # x = self.attn(self.norm1(x), batch_size=batch_size, H=H, W=W)[0]
            x = x + self.drop_path(self.attn(self.norm1(x), batch_size=batch_size, H=H, W=W)[0])
            attn = self.attn(self.norm1(x), batch_size=batch_size, H=H, W=W)[1]
            x = x + self.drop_path(self.mlp(self.norm2(x).view(batch_size, B_ // batch_size, -1, C)).view(B_, -1, C))
            return x, attn
        else:
            # x = self.attn(self.norm1(x), batch_size=batch_size, H=H, W=W)
            x = x + self.drop_path(self.attn(self.norm1(x), batch_size=batch_size, H=H, W=W))
            x = x + self.drop_path(self.mlp(self.norm2(x).view(batch_size, B_ // batch_size, -1, C)).view(B_, -1, C))
            return x

if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.autograd.Variable(torch.randn(4489, 85, 32)).to(device)
    gsa = Global_Scale_Attention([8, 4, 2, 1], attn_mask='Block').to(device)
    f = gsa(input, 1, H=67, W=67)

    num_params = 0
    for param in gsa.parameters():
        num_params += param.numel()
    print(num_params / 1e6)