import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils import *

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PyramidSqueeze(nn.Module):
    def __init__(self, dims_of_layers, dim=16, base_window_size=1, dropout=0.0, norm_layer=nn.LayerNorm):
        super(PyramidSqueeze, self).__init__()

        # params
        self.dim = dim
        self.feature_layers = dims_of_layers
        self.base_window_size = base_window_size
        self.size = [self.base_window_size * 2 ** (len(self.feature_layers) - 1 - i) for i in range(len(self.feature_layers))]

        # modules
        self.linears = nn.ModuleList([nn.Linear(self.feature_layers[i], self.dim) 
                                     for i in range(len(self.feature_layers))])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.feature_layers))])
        self.norm = nn.ModuleList([norm_layer(self.feature_layers[i]) for i in range(len(self.feature_layers))])
    
    def forward(self, multi_layer_features):
        assert len(multi_layer_features) == len(self.feature_layers)
        windowed_features = [window_partition(multi_layer_features[i], self.size[i]) for i in range(len(self.size))]
        normed_features = [self.norm[i](windowed_features[i]) for i in range(len(self.size))]
        squeezed_features = [self.dropouts[i](self.linears[i](normed_features[i])) for i in range(len(self.size))]
        return squeezed_features

class PyramidReverse(nn.Module):
    def __init__(self, dims_of_layers, base_pyramid_size, dim=16, base_window_size=1, dropout=0.0, drop_path=0.0, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PyramidReverse, self).__init__()

        # params
        self.dim = dim
        self.feature_layers = dims_of_layers
        self.base_pyramid_size = base_pyramid_size
        self.base_window_size = base_window_size
        self.size = [self.base_window_size * 2 ** (len(self.feature_layers) - 1 - i) for i in range(len(self.feature_layers))]

        # modules
        self.linears = nn.ModuleList([nn.Linear(self.dim, self.feature_layers[i]) 
                                     for i in range(len(self.feature_layers))])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.feature_layers))])
        self.norm = nn.ModuleList([norm_layer(self.dim) for i in range(len(self.feature_layers))])
        self.mlps = nn.ModuleList([Mlp(in_features=self.feature_layers[i], hidden_features=self.feature_layers[i] * mlp_ratio, 
                         act_layer=act_layer, drop=dropout) for i in range(len(self.feature_layers))])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, multi_pyramids):
        assert len(multi_pyramids) == len(self.feature_layers)
        reversed_features = [window_reverse(multi_pyramids[i], self.size[i], self.base_pyramid_size) for i in range(len(self.size))]
        normed_features = [self.norm[i](reversed_features[i]) for i in range(len(self.size))]
        proj_features = [self.dropouts[i](self.linears[i](normed_features[i])) for i in range(len(self.size))]
        f_features = [self.mlps[i](proj_features[i]) for i in range(len(self.size))]
        f_features = [(proj_features[i] + self.drop_path(f_features[i])).permute(0, 3, 1, 2).contiguous() for i in range(len(self.size))]
        return f_features

class Scale_Unified_Attention(nn.Module):
    def __init__(self, dims_of_layers, num_pyramid, num_heads=8, base_window_size=1, window_squeeze_drop=0.0, 
                 scale_attn_drop=0.0, spatial_attn_drop=0.0, window_reverse_drop=0.0, drop_path=0.0, mlp_ratio=4, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rpb=True, mask=None):
        super(Scale_Unified_Attention, self).__init__()

        # related params
        self.dim = dims_of_layers[0]
        self.num_heads = num_heads # num of attention head
        self.num_pyramid = num_pyramid # num of scale pyramids (equals to the num of pixels of the last feature map)
        self.base_pyramid_size = int(torch.sqrt(torch.tensor(self.num_pyramid))) # num of scale pyramids alongside )
        self.size = [base_window_size * 2 ** (len(dims_of_layers) - 1 - i) for i in range(len(dims_of_layers))] # window size
        self.layers = [self.dim // num_heads for i in range(len(dims_of_layers))] # num of feature layers of every attention head
        self.num_features = [self.size[i] ** 2 * self.layers[i] for i in range(len(dims_of_layers))]
        self.layers_per_head = sum(self.num_features)
        self.square_size = [self.size[i] ** 2  for i in range(len(self.size))]
        self.use_rpb = use_rpb
        self.mask = mask

        # modules
        self.pyramid_squeeze = PyramidSqueeze(dims_of_layers, dim=self.dim, base_window_size=base_window_size, norm_layer=norm_layer, dropout=window_squeeze_drop)
        self.pyramid_reverse = PyramidReverse(dims_of_layers, dim=self.dim, base_pyramid_size=self.base_pyramid_size, base_window_size=base_window_size, norm_layer=norm_layer, 
                                              dropout=window_reverse_drop, mlp_ratio=mlp_ratio, act_layer=act_layer)

        self.scale_qkv = nn.ModuleList([nn.Linear(self.dim, self.dim * 3) for i in range(len(dims_of_layers))])
        self.spatial_qkv = nn.Linear(sum(self.square_size) * self.dim // self.num_heads, 3)
        
        self.scale_softmax = nn.Softmax(dim=-1)
        self.scale_attn_drop = nn.Dropout(scale_attn_drop)
        self.spatial_softmax = nn.Softmax(dim=-1)
        self.spatial_attn_drop = nn.Dropout(spatial_attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # define a parameter table of relative scale/spatial bias

        self.relative_scale_bias_table = nn.Parameter(torch.zeros(2 * (sum(self.square_size) + sum(self.square_size[1:])) - 1, num_heads))
        # get pair-wise relative position index for each token inside the scale space
        coords_h_scale = torch.arange(sum(self.square_size))
        coords_w_scale = torch.arange(sum(self.square_size)).flip(0)
        offset = [0]
        for i in range(len(self.square_size) - 1):
            offset.append(sum(self.square_size[-i - 1:]))
        offset_h_scale = []
        offset_w_scale = []
        for i in range(len(self.square_size)):
            offset_w_scale.extend([offset[- i - 1] for _ in range(self.square_size[i])])
            offset_h_scale.extend([offset[i] for _ in range(self.square_size[i])])
        offset_h_scale = torch.tensor(offset_h_scale).unsqueeze(-1) * torch.ones((1, sum(self.square_size))).long()
        offset_w_scale = torch.tensor(offset_w_scale).unsqueeze(0) * torch.ones((sum(self.square_size), 1)).long()
        coords_scale = torch.stack(torch.meshgrid([coords_h_scale, coords_w_scale]))  # 2, len_code, len_code
        relative_scale_index = coords_scale.sum(0) + offset_h_scale + offset_w_scale  # len_code, len_code
        self.register_buffer("relative_scale_index", relative_scale_index)
        trunc_normal_(self.relative_scale_bias_table, std=.02)

        self.relative_spatial_bias_table = nn.Parameter(
            torch.zeros((2 * self.base_pyramid_size - 1) * (2 * self.base_pyramid_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the spatial dim
        coords_h_spatial = torch.arange(self.base_pyramid_size)
        coords_w_spatial = torch.arange(self.base_pyramid_size)
        coords_spatial = torch.stack(torch.meshgrid([coords_h_spatial, coords_w_spatial]))  # 2, Wh, Ww
        coords_spatial_flatten = torch.flatten(coords_spatial, 1)  # 2, Wh*Ww
        relative_coords_spatial = coords_spatial_flatten[:, :, None] - coords_spatial_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords_spatial = relative_coords_spatial.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords_spatial[:, :, 0] += self.base_pyramid_size - 1  # shift to start from 0
        relative_coords_spatial[:, :, 1] += self.base_pyramid_size - 1
        relative_coords_spatial[:, :, 0] *= 2 * self.base_pyramid_size - 1
        relative_spatial_index = relative_coords_spatial.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_spatial_index", relative_spatial_index)
        trunc_normal_(self.relative_spatial_bias_table, std=.02)

    def forward(self, multi_layer_features):
        # assert (multi_layer_features[-1].size(-2) / self.size[-1]) == self.base_pyramid_size
        B = multi_layer_features[0].size(0)
        device = multi_layer_features[0].device
        squeezed_features = self.pyramid_squeeze(multi_layer_features)

        '''
        scale attention
        '''
        scale_q, scale_k, scale_v = [], [], []
        scale_space_num = 0
        for i in range(len(squeezed_features)):
            B_, N, C = squeezed_features[i].shape
            if i == 0:
                scale_space_num = B_
            else:
                assert scale_space_num == B_
            qkv = self.scale_qkv[i](squeezed_features[i]).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)
            scale_q.append(qkv[0])
            scale_k.append(qkv[1])
            scale_v.append(qkv[2])
        scale_q = torch.cat(scale_q, dim=-2)
        scale_k = torch.cat(scale_k, dim=-2)
        scale_v = torch.cat(scale_v, dim=-2)
        regularize_scale = sum(self.square_size) ** -0.5
        
        scale_q = scale_q * regularize_scale
        scale_attn = (scale_q @ scale_k.transpose(-2, -1))

        # build relative scale bias
        if self.use_rpb:
            relative_scale_bias = self.relative_scale_bias_table[self.relative_scale_index.view(-1)].view(
                                                                sum(self.square_size), sum(self.square_size), -1)  # Wh*Ww,Wh*Ww,nH
            relative_scale_bias = relative_scale_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            scale_attn = scale_attn + relative_scale_bias.unsqueeze(0)
        
        # build scale mask
        if self.mask is not None:
            scale_mask = create_scale_mask([self.square_size[i] for i in range(len(self.square_size))], [1 for _ in range(len(self.square_size))], -100, self.mask).to(device)
            scale_mask.requires_grad = False
            scale_attn = scale_attn + scale_mask
        
        scale_attn = self.scale_softmax(scale_attn) # num_pyramid*B, num_heads, layers_per_head, layers_per_head
        scale_attn = self.scale_attn_drop(scale_attn)

        b = multi_layer_features[-1].size(-2) // self.size[-1]
        n = b ** 2

        scale_x = (scale_attn @ scale_v).view(B, n, self.num_heads, -1)

        '''
        spatial attention(aggregation)
        '''
        spatial_qkv = self.spatial_qkv(scale_x).permute(3, 0, 2, 1)
        spatial_q, spatial_k, spatial_v = spatial_qkv[0].unsqueeze(-1), spatial_qkv[1].unsqueeze(-1), spatial_qkv[2].unsqueeze(-1)
        regularize_spatial = n ** -0.5

        spatial_q = spatial_q * regularize_spatial
        spatial_attn = (spatial_q @ spatial_k.transpose(-2, -1))

        # build relative scale bias
        if self.use_rpb:
            size = int(multi_layer_features[-1].size(-2) / self.size[-1])

            coords_h_spatial = torch.arange(size)
            coords_w_spatial = torch.arange(size)
            coords_spatial = torch.stack(torch.meshgrid([coords_h_spatial, coords_w_spatial]))  # 2, Wh, Ww
            coords_spatial_flatten = torch.flatten(coords_spatial, 1)  # 2, Wh*Ww
            relative_coords_spatial = coords_spatial_flatten[:, :, None] - coords_spatial_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords_spatial = relative_coords_spatial.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords_spatial[:, :, 0] += size - 1  # shift to start from 0
            relative_coords_spatial[:, :, 1] += size - 1
            relative_coords_spatial[:, :, 0] *= 2 * size - 1
            relative_spatial_index = relative_coords_spatial.sum(-1)  # Wh*Ww, Wh*Ww

            if size != self.base_pyramid_size:
                relative_spatial_bias_table = F.interpolate(self.relative_spatial_bias_table.unsqueeze(0).transpose(1, 2), 
                                                      size=(2 * size - 1) * (2 * size - 1), mode='linear').squeeze().transpose(0, 1)
                relative_spatial_bias = relative_spatial_bias_table[relative_spatial_index.view(-1)].view(
                    size ** 2, size ** 2, -1)  # Wh*Ww,Wh*Ww,nH
            else:
                relative_spatial_bias = self.relative_spatial_bias_table[relative_spatial_index.view(-1)].view(
                    size ** 2, size ** 2, -1)  # Wh*Ww,Wh*Ww,nH
            relative_spatial_bias = relative_spatial_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            # if multi_layer_features[-1].size(-2) / self.size[-1] != self.base_pyramid_size:
            #     size = int(multi_layer_features[-1].size(-2) / self.size[-1]) ** 2
            #     relative_spatial_bias = F.interpolate(relative_spatial_bias.unsqueeze(0), size=(size, size), mode='bilinear').squeeze()

            spatial_attn = spatial_attn + relative_spatial_bias.unsqueeze(0)

        if self.mask is not None:
            spatial_mask = create_spatial_mask(n, -100).to(device)
            spatial_mask.requires_grad = False
            spatial_attn = spatial_attn + spatial_mask
        
        spatial_attn = self.spatial_softmax(spatial_attn) # B, num_heads, num_pyramid, num_pyramid
        spatial_attn = self.spatial_attn_drop(spatial_attn)
        spatial_x = (spatial_attn @ spatial_v).permute(0, 2, 1, 3).contiguous()

        ss_x = (self.drop_path(scale_x * spatial_x) + scale_x).view(-1, self.num_heads, sum(self.square_size), self.dim // self.num_heads)

        start_point = [0]
        for i in range(len(self.num_features) - 1):
            start_point.append(self.square_size[i])
        for i in range(len(start_point)):
            if i > 0:
                start_point[i] += start_point[i - 1]
        
        attn_features = []
        for i in range(len(self.square_size)):
            if i < len(self.square_size) - 1:
                temp = ss_x[:, :, start_point[i] : start_point[i + 1]].view(scale_space_num, 
                         self.num_heads, self.square_size[i], -1).permute(0, 2, 1, 3).flatten(2).contiguous()
            else:
                temp = ss_x[:, :, start_point[-1]:].view(scale_space_num, 
                         self.num_heads, self.square_size[i], -1).permute(0, 2, 1, 3).flatten(2).contiguous()
            attn_features.append(temp)
        
        proj_features = self.pyramid_reverse(attn_features)

        return proj_features

if __name__ == '__main__':
    device = torch.device('cpu')
    # l1 = torch.autograd.Variable(torch.randn(2, 64, 128, 128)).to(device)
    # l2 = torch.autograd.Variable(torch.randn(2, 128, 64, 64)).to(device)
    # l3 = torch.autograd.Variable(torch.randn(2, 256, 32, 32)).to(device)
    # l4 = torch.autograd.Variable(torch.randn(2, 512, 16, 16)).to(device)

    l1 = torch.autograd.Variable(torch.randn(2, 64, 256, 256)).to(device)
    l2 = torch.autograd.Variable(torch.randn(2, 128, 128, 128)).to(device)
    l3 = torch.autograd.Variable(torch.randn(2, 256, 64, 64)).to(device)
    l4 = torch.autograd.Variable(torch.randn(2, 512, 32, 32)).to(device)

    features = [l1, l2, l3, l4]
    filters = [features[i].size(1) for i in range(len(features))]
    num_pyramid = features[-1].size(-2) * features[-1].size(-1)
    sua = Scale_Unified_Attention(filters, 256, mask='Block').to(device)
    f = sua(features)

    num_params = 0
    for param in sua.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    print(f[0].size())
    print(f[1].size())
    print(f[2].size())
    print(f[3].size())