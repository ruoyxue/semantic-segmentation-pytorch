import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGN_CORNERS = True

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer, act_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer, act_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        relu = act_layer()
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class UperNet_Decoder(nn.Module):
    def __init__(self, filters, out_ch, dim=512, ppm_size=(1, 2, 3, 6)):
        super(UperNet_Decoder, self).__init__()
        self.dim = dim
        self.psp = PSPModule(filters[-1], self.dim, sizes=ppm_size)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in filters[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.dim, 3, padding=1),
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
        
        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(len(filters) * self.dim, self.dim, 3, padding=1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )
        
        self.cls_seg = nn.Conv2d(self.dim, out_ch, kernel_size=1)
        
    def forward(self, x):

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp(x[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=ALIGN_CORNERS)
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=ALIGN_CORNERS)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output

class UperNet_OCR_Decoder(nn.Module):
    def __init__(self, filters, dim=512, ppm_size=(1, 2, 3, 6)):
        super(UperNet_OCR_Decoder, self).__init__()
        self.dim = dim
        self.psp = PSPModule(filters[-1], self.dim, sizes=ppm_size)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in filters[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.dim, 3, padding=1),
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

        self.last_output_chs = len(filters) * self.dim
        
    def forward(self, x):

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp(x[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=ALIGN_CORNERS)
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=ALIGN_CORNERS)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return fpn_outs