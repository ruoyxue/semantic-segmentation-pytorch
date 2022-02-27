import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .utils import *
from .modules import *
from .local_scale_attention import Local_Scale_Unification_Block
from .global_scale_attention import Global_Scale_Unification_Block

class Scale_Unified_Attention(nn.Module):
    def __init__(self, 
                 dims_of_layers, 
                 dim=64, 
                 num_heads=16, 
                 base_window_size=1, 
                 window_squeeze_drop=0.0, 
                 window_reverse_drop=0.0,
                 scale_attn_drop=0.0, 
                 scale_proj_drop=0.0,
                 spatial_attn_drop=0.0, 
                 spatial_proj_drop=0.0,
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_path=0.0, 
                 mlp_ratio=8, 
                 lspe_k_size=7, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 use_rsb=True, 
                 use_lspe=True, 
                 scale_mask=None,
                 spatial_mask=None,
                 vis=False):
        super(Scale_Unified_Attention, self).__init__()

        # related params
        self.dim = dim
        self.num_heads = num_heads # num of attention head
        self.size = [base_window_size * 2 ** (len(dims_of_layers) - 1 - i) for i in range(len(dims_of_layers))] # window size
        self.square_size = [self.size[i] ** 2  for i in range(len(self.size))]
        self.use_rsb = use_rsb
        self.use_lspe = use_lspe
        self.scale_mask = scale_mask
        self.spatial_mask = spatial_mask
        self.vis = vis

        # modules
        self.pyramid_squeeze = PyramidSqueeze(dims_of_layers, dim=self.dim, base_window_size=base_window_size, norm_layer=norm_layer, dropout=window_squeeze_drop)
        self.pyramid_reverse = PyramidReverse(dims_of_layers, dim=self.dim, base_window_size=base_window_size, norm_layer=norm_layer, dropout=window_reverse_drop)
        self.LSA = Local_Scale_Unification_Block(self.size, 
                                                dim=self.dim, 
                                                num_heads=self.num_heads, 
                                                mlp_ratio=mlp_ratio,
                                                scale_attn_drop=scale_attn_drop, 
                                                scale_proj_drop=scale_proj_drop,
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale,
                                                drop_path=drop_path,
                                                act_layer=act_layer, 
                                                norm_layer=norm_layer,
                                                use_rsb=self.use_rsb, 
                                                attn_mask=self.scale_mask,
                                                vis=self.vis)
        self.GSA = Global_Scale_Unification_Block(self.size, 
                                                dim=self.dim, 
                                                num_heads=self.num_heads, 
                                                mlp_ratio=mlp_ratio,
                                                spatial_attn_drop=spatial_attn_drop, 
                                                spatial_proj_drop=spatial_proj_drop,
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale,
                                                drop_path=drop_path,
                                                act_layer=act_layer, 
                                                norm_layer=norm_layer,
                                                use_lspe=self.use_lspe, 
                                                lspe_k_size=lspe_k_size,
                                                attn_mask=self.spatial_mask,
                                                vis=self.vis)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.init_weights()
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            pretrained_dict = torch.load(pretrained, map_location={'cuda': 'cpu'})
            model_dict = self.state_dict()
            # pretrained_dict['model'].pop('norm.weight')
            # pretrained_dict['model'].pop('norm.bias')
            # pretrained_dict['model'].pop('head.weight')
            # pretrained_dict['model'].pop('head.bias')
            # for k, v in list(pretrained_dict['model'].items()):
            #     if str.find(k, 'relative_position') != -1:
            #         pretrained_dict['model'].pop(k)
            # pretrained_dict_new = {k: v for k, v in pretrained_dict['model'].items()
            #                    if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, multi_layer_features):

        # padding for window split
        original_feature_sizes = []
        num_window_h = math.ceil(multi_layer_features[-1].size(-2) / float(self.size[-1]))
        num_window_w = math.ceil(multi_layer_features[-1].size(-1) / float(self.size[-1]))
        for i in range(len(multi_layer_features)):
            original_feature_sizes.append(multi_layer_features[i].size())
            bottom_padding = num_window_h * self.size[i] - multi_layer_features[i].size(-2)
            right_padding = num_window_w * self.size[i] - multi_layer_features[i].size(-1)
            multi_layer_features[i] = F.pad(multi_layer_features[i], (0, right_padding, 0, bottom_padding), "constant", 0)

        # pyramid squeeze
        B = multi_layer_features[0].size(0)
        x = self.pyramid_squeeze(multi_layer_features)
        x = torch.cat(x, dim=1)
        base_pyramid_size = int(math.sqrt(x.size(0) // B))

        # scale attention
        if self.vis:
            x, l_attn = self.LSA(x)
        else:
            x = self.LSA(x)

        # spatial attention(aggregation)
        if self.vis:
            x, g_attn = self.GSA(x, B)
        else:
            x = self.GSA(x, B)

        # pyramid reverse
        x = torch.split(x, self.square_size, dim=1)
        
        x = self.pyramid_reverse(x, base_pyramid_size)

        # identity
        # x = [multi_layer_features[i] + self.drop_path(torch.sigmoid(x[i]) * multi_layer_features[i]) for i in range(len(multi_layer_features))]
        # x = [x[i] * multi_layer_features[i] for i in range(len(multi_layer_features))]
        x = [multi_layer_features[i] + self.drop_path(x[i] * multi_layer_features[i]) for i in range(len(multi_layer_features))]
        x = [x[i][:, :, 0:original_feature_sizes[i][-2], 0:original_feature_sizes[i][-1]]  for i in range(len(multi_layer_features))]
        # x = [multi_layer_features[i][:, :, 0:original_feature_sizes[i][-2], 0:original_feature_sizes[i][-1]]  for i in range(len(multi_layer_features))]

        if self.vis:
            return x, l_attn, g_attn
        else:
            return x

if __name__ == '__main__':
    device = torch.device('cpu')
    # l1 = torch.autograd.Variable(torch.randn(2, 64, 128, 128)).to(device)
    # l2 = torch.autograd.Variable(torch.randn(2, 128, 64, 64)).to(device)
    # l3 = torch.autograd.Variable(torch.randn(2, 256, 32, 32)).to(device)
    # l4 = torch.autograd.Variable(torch.randn(2, 512, 16, 16)).to(device)
    x = [533]
    for i in range(3):
        y = math.ceil(x[-1] / 2.0)
        x.append(y)
    print(x)

    l1 = torch.autograd.Variable(torch.randn(1, 64, x[0], x[0])).to(device)
    l2 = torch.autograd.Variable(torch.randn(1, 128, x[1], x[1])).to(device)
    l3 = torch.autograd.Variable(torch.randn(1, 256, x[2], x[2])).to(device)
    l4 = torch.autograd.Variable(torch.randn(1, 512, x[3], x[3])).to(device)

    features = [l1, l2, l3, l4]
    filters = [features[i].size(1) for i in range(len(features))]
    num_pyramid = features[-1].size(-2) * features[-1].size(-1)
    sua = Scale_Unified_Attention(filters).to(device)
    f = sua(features)

    num_params = 0
    for param in sua.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    print(f[0].size())
    print(f[1].size())
    print(f[2].size())
    print(f[3].size())