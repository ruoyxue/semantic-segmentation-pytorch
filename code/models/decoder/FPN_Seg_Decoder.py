from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class Panoptic_FPN_Decoder(nn.Module):
    def __init__(self, filters, n_classes, dim=256):
        super(Panoptic_FPN_Decoder, self).__init__()
        assert min(filters) == filters[0]
        self.filters = filters
        self.dim = dim

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.filters)):
            head_length = max(
                1,
                int(np.log2(self.filters[i]) - np.log2(self.filters[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(
                        nn.Conv2d(self.filters[i] if k == 0 else self.dim, self.dim, 3, padding=1),
                        nn.BatchNorm2d(self.dim),
                        nn.ReLU(inplace=True)
                    )
                )
                if self.filters[i] != self.filters[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=ALIGN_CORNERS))
            self.scale_heads.append(nn.Sequential(*scale_head))
        
        self.cls_seg = nn.Conv2d(self.dim, n_classes, kernel_size=1)
    
    def forward(self, x):
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.filters)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=ALIGN_CORNERS)

        output = self.cls_seg(output)
        return output

class Vanilla_FPN_Decoder(nn.Module):
    def __init__(self, filters, n_classes, dim=256):
        super(Vanilla_FPN_Decoder, self).__init__()
        self.reduce1 = nn.Conv2d(filters[3], dim, 1, 1, 0)
        self.reduce2 = nn.Conv2d(filters[2], dim, 1, 1, 0)
        self.reduce3 = nn.Conv2d(filters[1], dim, 1, 1, 0)
        self.reduce4 = nn.Conv2d(filters[0], dim, 1, 1, 0)
        self.last = nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=dim,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )
    
    def forward(self, x):
        out = self.reduce1(x[3])
        out = self.reduce2(x[2]) + F.interpolate(out, size=(x[2].size(-2), x[2].size(-1)), mode='bilinear', align_corners=True)
        out = self.reduce3(x[1]) + F.interpolate(out, size=(x[1].size(-2), x[1].size(-1)), mode='bilinear', align_corners=True)
        out = self.reduce4(x[0]) + F.interpolate(out, size=(x[0].size(-2), x[0].size(-1)), mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=(x[0].size(-2) * 4, x[0].size(-1) * 4) , mode='bilinear', align_corners=True)
        out = self.last(out)

        return out

class HRNet_FPN_Seg_OCR_Decoder(nn.Module):

    def __init__(self):
        super(HRNet_FPN_Seg_OCR_Decoder, self).__init__()
    
    def forward(self, x):
        assert len(x) == 4
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        return feats

class HRNet_FPN_Seg_Decoder(nn.Module):

    def __init__(self, last_inp_channels, n_classes):
        super(HRNet_FPN_Seg_Decoder, self).__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )
    
    def forward(self, x):
        assert len(x) == 4
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        out = self.last_layer(feats)
        return out