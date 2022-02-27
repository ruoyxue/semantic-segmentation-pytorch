from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from code.models.backbone.ResNet_Dilated import ResNet_Dilated
from code.lufangxiao.module.ASPP import _ASPP
from code.models.decoder.Deeplabv3p_Decoder import DeepLabv3p_decoder


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, in_ch, n_classes, atrous_rates, multi_grids, output_stride, backbone):
        super(DeepLabV3Plus, self).__init__()

        if backbone == 'resnet50' or backbone == 'resnet101':
            mid_ch = 256
            encoder_ch = 2048
        else:
            raise ValueError("wrong indicator for backbone, which should be selected from {'resnet50', 'resnet101'}")

        self.backbone = ResNet_Dilated(in_ch, multi_grids, output_stride, backbone)
        self.aspp = _ASPP(encoder_ch, mid_ch, atrous_rates)

        # Decoder
        reduce_ch = 48
        self.decoder = DeepLabv3p_decoder(mid_ch, reduce_ch, n_classes)

        self.name = "DeepLabv3plus"
    
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

    def forward(self, x):
        b = self.backbone(x)
        h = self.aspp(b[3])
        encoder_outs = []
        encoder_outs.append(b[0])
        encoder_outs.append(h)
        h = self.decoder(encoder_outs)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return h


if __name__ == "__main__":
    model = DeepLabV3Plus(
        in_ch=3,
        n_classes=21,
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
        backbone='resnet101'
    )
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    model.init_weights('/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth')
    # pretrained_dict = torch.load('/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth')
    # for k, v in pretrained_dict.items():
    #     print(k)
    # pretrained_dict.pop('fc.weight')
    # pretrained_dict.pop('fc.bias')
    # model.load_state_dict(pretrained_dict, strict=False)
    
    # for k, v in model.state_dict().items():
    #     print(k)

    # print(model)
    # print("input:", image.shape)
    print("output:", model(image).shape)