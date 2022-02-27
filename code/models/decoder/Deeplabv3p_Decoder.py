from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from code.models.backbone.ResNet_Dilated import _ConvBnReLU

from collections import OrderedDict

class DeepLabv3p_decoder(nn.Module):

    def __init__(self, mid_ch, reduce_ch, n_classes):
        super(DeepLabv3p_decoder, self).__init__()

        self.reduce = _ConvBnReLU(mid_ch, reduce_ch, 1, 1, 0, 1)
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(mid_ch + reduce_ch, mid_ch, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(mid_ch, mid_ch, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(mid_ch, n_classes, kernel_size=1)),
                ]
            )
        )
    
    def forward(self, encoder_outs):
        assert len(encoder_outs) == 2
        mid_feature = encoder_outs[0]
        deep_feature = encoder_outs[1]
        m = self.reduce(mid_feature)
        x = F.interpolate(deep_feature, size=m.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat((x, m), dim=1)
        h = self.fc(h)
        return h