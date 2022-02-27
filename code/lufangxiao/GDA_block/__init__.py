import torch
import torch.nn as nn
import yaml
from .SUA_v3 import Scale_Unified_Attention

def build_sua_module(dims_of_layers, cfg_filepath):
    sua_cfg_file = open(cfg_filepath, 'r')
    sua_config = yaml.load(sua_cfg_file, Loader=yaml.FullLoader)
    sua_cfg_file.close()

    if sua_config['ACT_LAYER'] == 'GELU':
        act_layer = nn.GELU
    else:
        raise ValueError("This script does not support Activation Layer exclude 'GELU'.")
    
    if sua_config['NORM_LAYER'] == 'LayerNorm':
        norm_layer = nn.LayerNorm
    else:
        raise ValueError("This script does not support Norm Layer exclude 'LayerNorm'.")
    
    if sua_config['SCALE_MASK'] == 'None':
        scale_mask = None
    elif sua_config['SCALE_MASK'] in ['Self', 'Block', 'Layer']:
        scale_mask = sua_config['SCALE_MASK']
    else:
        raise ValueError("Invalid indicator for 'scale_mask', which should be 'Self', 'Block' or 'Layer'.")

    if sua_config['SPATIAL_MASK'] == 'None':
        spatial_mask = None
    elif sua_config['SPATIAL_MASK'] in ['Self', 'Block', 'Layer']:
        spatial_mask = sua_config['SPATIAL_MASK']
    else:
        raise ValueError("Invalid indicator for 'spatial_mask', which should be 'Self', 'Block' or 'Layer'.")
    
    if sua_config['QK_SCALE'] == 'None':
        qk_scale = None
    else:
        qk_scale = sua_config['QK_SCALE']

    module = Scale_Unified_Attention(dims_of_layers, 
                                     dim=sua_config['DIM'],
                                     num_heads=sua_config['NUM_HEADS'],
                                     base_window_size=sua_config['BASE_WINDOW_SIZE'],
                                     window_squeeze_drop=sua_config['WINDOW_SQUEEZE_DROP'],
                                     window_reverse_drop=sua_config['WINDOW_REVERSE_DROP'],
                                     scale_attn_drop=sua_config['SCALE_ATTN_DROP'],
                                     scale_proj_drop=sua_config['SCALE_PROJ_DROP'],
                                     spatial_attn_drop=sua_config['SPATIAL_ATTN_DROP'],
                                     spatial_proj_drop=sua_config['SPATIAL_PROJ_DROP'],
                                     drop_path=sua_config['DROP_PATH'],
                                     mlp_ratio=sua_config['MLP_RATIO'],
                                     qkv_bias=sua_config['QKV_BIAS'],
                                     qk_scale=qk_scale,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     use_rsb=sua_config['USE_RSB'], 
                                     use_lspe=sua_config['USE_LSPE'],
                                     lspe_k_size=sua_config['LSPE_K_SIZE'],
                                     scale_mask=scale_mask,
                                     spatial_mask=spatial_mask,
                                     vis=sua_config['VIS'])
    
    return module