import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGN_CORNERS = True

class FPN_Neck(nn.Module):
    def __init__(self, filters, dim):
        super(FPN_Neck, self).__init__()
        self.filters = filters
        self.dim = dim

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(self.filters)):
            l_conv = nn.Sequential(
                        nn.Conv2d(filters[i], self.dim, 1),
                        nn.BatchNorm2d(self.dim),
                        nn.ReLU()
                    )
            fpn_conv = nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, 3, padding=1),
                        nn.BatchNorm2d(self.dim),
                        nn.ReLU()
                    )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
    def get_filters(self):
        return [self.dim for _ in range(len(self.filters))]
    
    def forward(self, x):
        assert len(x) == len(self.filters)

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, mode='bilinear', align_corners=ALIGN_CORNERS)
        
        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return outs