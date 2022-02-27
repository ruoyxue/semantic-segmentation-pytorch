# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

import os

import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .backbone import HighResolutionNet
from .decoder.FPN_Seg_Decoder import HRNet_FPN_Seg_Decoder

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class SegHRNet(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32', pretrained=False):
        super().__init__()
        self.pretrained_path = ""
        self.backbone = backbone
        if backbone not in ["hr-w18", "hr-w32", "hr-w48"]:
            raise ValueError("model gets invalid backbone, expects in [hr-w18, hr-w32, hr-w48]")
        if self.backbone == "hr-w18":
            self.pretrained_path = "models/backbone/pretrained/hrnetv2_w18_imagenet_pretrained.pth"
        elif self.backbone == "hr-w32":
            self.pretrained_path = "models/backbone/pretrained/hrnetv2_w32_imagenet_pretrained.pth"
        elif self.backbone == "hr-w48":
            self.pretrained_path = "models/backbone/pretrained/hrnetv2_w48_imagenet_pretrained.pth"

        self.backbone = HighResolutionNet(in_ch, backbone=backbone)
        self.decoder = HRNet_FPN_Seg_Decoder(self.backbone.last_inp_channels, n_classes)
        self.init_weights(pretrained=pretrained)
    
    def init_weights(self, pretrained):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrained:
            self.backbone.init_weights(pretrained=self.pretrained_path)

    def forward(self, input):
        x = self.backbone(input)
        x = self.decoder(x)
        output = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=ALIGN_CORNERS)
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512)).to(device)
    net = SegHRNet(in_ch=3, n_classes=2).to(device)
    print(net(input).size())
