import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .utils import *

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_lspe=False, lspe_k_size=7):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.use_lspe = use_lspe
        if self.use_lspe:
            self.dwconv = DWConv(lspe_k_size, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        if self.use_lspe:
            x = x + self.dwconv(x, H, W)
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
        squeezed_features = [self.dropouts[i](self.linears[i](self.norm[i](multi_layer_features[i].permute(0, 2, 3, 1)))) for i in range(len(self.size))]
        windowed_features = [window_partition(squeezed_features[i], self.size[i]) for i in range(len(self.size))]
        return windowed_features

class PyramidReverse(nn.Module):
    def __init__(self, dims_of_layers, dim=16, base_window_size=1, dropout=0.0, norm_layer=nn.LayerNorm):
        super(PyramidReverse, self).__init__()

        # params
        self.dim = dim
        self.feature_layers = dims_of_layers
        self.base_window_size = base_window_size
        self.size = [self.base_window_size * 2 ** (len(self.feature_layers) - 1 - i) for i in range(len(self.feature_layers))]

        # modules
        self.linears = nn.ModuleList([nn.Linear(self.dim, self.feature_layers[i]) 
                                     for i in range(len(self.feature_layers))])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.feature_layers))])
        self.norm = nn.ModuleList([norm_layer(self.dim) for i in range(len(self.feature_layers))])
    
    def forward(self, multi_pyramids, base_pyramid_size):
        assert len(multi_pyramids) == len(self.feature_layers)
        reversed_features = [window_reverse(multi_pyramids[i], self.size[i], base_pyramid_size) for i in range(len(self.size))]
        proj_features = [self.dropouts[i](self.linears[i](self.norm[i](reversed_features[i]))).permute(0, 3, 1, 2) for i in range(len(self.size))]
        return proj_features

class DWConv(nn.Module):
    def __init__(self, k_size=3, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, k_size, 1, k_size // 2, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, M, C = x.shape
        assert N == H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B * M, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).view(B, M, C, N).permute(0, 3, 1, 2)

        return x

class MultiScale_DWConv(nn.Module):
    def __init__(self, k_sizes, dim_in, dim_out):
        super(MultiScale_DWConv, self).__init__()
        self.k_sizes = k_sizes
        self.lspe_dwconvs = nn.ModuleList([nn.Conv2d(dim_in, dim_out, self.k_sizes[i], 1, self.k_sizes[i] // 2, 
                                           bias=True, groups=dim_in) for i in range(len(self.k_sizes))])

    def forward(self, x):
        out = self.lspe_dwconvs[0](x)
        for i in range(len(self.k_sizes) - 1):
            out += self.lspe_dwconvs[i + 1](x)
        return out

class Scale_AvgPooling(nn.Module):
    def __init__(self, size):
        super(Scale_AvgPooling, self).__init__()
        self.size = size
        self.square_size = [self.size[i] ** 2  for i in range(len(self.size))]
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        '''
        input: B_ * sum(square_sizes) * dim
        output: B_ * len(square_sizes) * dim
        '''
        assert x.size(-2) == sum(self.square_size)
        B_, _, C = x.shape

        x = torch.split(x, self.square_size, dim=1)
        x = [self.avgpooling(x[i].transpose(1, 2).view(B_, C, self.size[i], self.size[i])).flatten(2) for i in range(len(x))]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        return x

class Scale_MaxPooling(nn.Module):
    def __init__(self, size):
        super(Scale_MaxPooling, self).__init__()
        self.size = size
        self.square_size = [self.size[i] ** 2  for i in range(len(self.size))]
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        '''
        input: B_ * sum(square_sizes) * dim
        output: B_ * len(square_sizes) * dim
        '''
        assert x.size(-2) == sum(self.square_size)
        B_, _, C = x.shape

        x = torch.split(x, self.square_size, dim=1)
        x = [self.maxpooling(x[i].transpose(1, 2).view(B_, C, self.size[i], self.size[i])).flatten(2) for i in range(len(x))]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        return x

class Scale_UnPooling(nn.Module):
    def __init__(self, size):
        super(Scale_UnPooling, self).__init__()
        self.size = size
        self.square_size = [self.size[i] ** 2  for i in range(len(self.size))]
    
    def forward(self, x):
        '''
        input: B_ * len(square_sizes) * dim
        output: B_ * sum(square_sizes) * dim
        '''
        x = x.transpose(1, 2)
        out = [x[:, :, i].unsqueeze(-1).repeat(1, 1, self.square_size[i]) for i in range(len(self.size))]
        x = torch.cat(out, dim=-1).transpose(1, 2)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')
    input = torch.autograd.Variable(torch.randn(4489, 85, 32)).to(device)
    sap = Scale_AvgPooling([8, 4, 2, 1]).to(device)
    p = sap(input)
    print(p.size())