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
import torch.nn.functional as F

from code.models.backbone import HighResolutionNet
from code.lufangxiao.module.OCR import OCR_Module
from code.models.decoder.FPN_Seg_Decoder import HRNet_FPN_Seg_OCR_Decoder

class SegHRNet_OCR(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32'):
        super(SegHRNet_OCR, self).__init__()
        self.backbone = HighResolutionNet(in_ch, backbone=backbone)
        self.decoder = HRNet_FPN_Seg_OCR_Decoder()
        self.ocr = OCR_Module(n_classes, self.backbone.last_inp_channels)
    
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
        feats = self.decoder(x)
        out = self.ocr(feats)
        out = [F.interpolate(o, size=input.shape[2:], mode='bilinear', align_corners=True) for o in out]
        return out


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512)).to(device)
    net = SegHRNet_OCR(3, 17).to(device)
    print(net(input)[0].size())