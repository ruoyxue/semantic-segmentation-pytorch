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

from code.models.backbone import HighResolutionNet
from code.models.decoder.LinkNet_Decoder import LinkNet_Decoder

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class SegHR_LinkNet(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32'):
        super(SegHR_LinkNet, self).__init__()
        self.backbone = HighResolutionNet(in_ch, backbone=backbone)
        filters = self.backbone.get_filters()
        self.decoder = LinkNet_Decoder(filters, n_classes)
        self.name = "LinkNet_HRNet"
    
    def init_weights(self, pretrained=''):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            self.backbone.init_weights(pretrained=pretrained)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

    def forward(self, input):
        x = self.backbone(input)
        output = self.decoder(x)
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 1500, 1500)).to(device)
    net = SegHR_LinkNet(3, 17).to(device)
    print(net(input).size())