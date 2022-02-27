# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch.nn as nn
import torch._utils

from .backbone import HighResolutionNet
from .decoder.FPN_Seg_Decoder import Vanilla_FPN_Decoder

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class SegHRNet_DA(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32', pretrained=False):
        super(SegHRNet_DA, self).__init__()
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
        filters = self.backbone.get_filters()
        self.decoder = Vanilla_FPN_Decoder(filters, n_classes)
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
        out = self.decoder(x)
        return out
