"""
FCNs: FCN_8s, FCN_16s, FCN_32s
based on VGG16

Reference: https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
"""

import torch.nn as nn
import torch
from typing import Union


class FCN(nn.Module):
    """ Base class for different up-sampling strategy FCNs
    :param n_class: label categories
    """
    def __init__(self, n_class: int = 2):
        super().__init__()
        self.maxpooling_block1 = nn.Sequential(
            _FCNConvBlock(3, 64),
            _FCNConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.maxpooling_block2 = nn.Sequential(
            _FCNConvBlock(64, 128),
            _FCNConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.maxpooling_block3 = nn.Sequential(
            _FCNConvBlock(128, 256),
            _FCNConvBlock(256, 256),
            _FCNConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.maxpooling_block4 = nn.Sequential(
            _FCNConvBlock(256, 512),
            _FCNConvBlock(512, 512),
            _FCNConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.maxpooling_block5 = nn.Sequential(
            _FCNConvBlock(512, 512),
            _FCNConvBlock(512, 512),
            _FCNConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.upsample1 = _FCNDeconvBlock(512, 512)
        self.upsample2 = _FCNDeconvBlock(512, 256)
        self.upsample3 = _FCNDeconvBlock(256, 128)
        self.upsample4 = _FCNDeconvBlock(128, 64)
        self.upsample5 = _FCNDeconvBlock(64, 32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self._initialisation()

    def _initialisation(self):
        """ weight and bias initialisation """
        def init_weight(layer):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.maxpooling_block1.apply(init_weight)
        self.maxpooling_block2.apply(init_weight)
        self.maxpooling_block3.apply(init_weight)
        self.maxpooling_block4.apply(init_weight)
        self.maxpooling_block5.apply(init_weight)
        self.upsample1.apply(init_weight)
        self.upsample2.apply(init_weight)
        self.upsample3.apply(init_weight)
        self.upsample4.apply(init_weight)
        self.upsample5.apply(init_weight)
        self.classifier.apply(init_weight)

    def forward(self, x):
        raise NotImplementedError


class FCN32s(FCN):
    def __init__(self, n_class: int = 2):
        super().__init__(n_class)

    def forward(self, x):
        x1 = self.maxpooling_block1(x)
        x2 = self.maxpooling_block2(x1)
        x3 = self.maxpooling_block3(x2)
        x4 = self.maxpooling_block4(x3)
        x5 = self.maxpooling_block5(x4)
        output = self.upsample1(x5)
        output = self.upsample2(output)
        output = self.upsample3(output)
        output = self.upsample4(output)
        output = self.upsample5(output)
        output = self.classifier(output)
        return output


class FCN16s(FCN):
    def __init__(self, n_class: int = 2):
        super().__init__(n_class)
        del self.upsample1
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        nn.init.kaiming_normal_(self.deconv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.deconv1.bias)

    def forward(self, x):
        x1 = self.maxpooling_block1(x)
        x2 = self.maxpooling_block2(x1)
        x3 = self.maxpooling_block3(x2)
        x4 = self.maxpooling_block4(x3)
        x5 = self.maxpooling_block5(x4)
        # we have to ensure both part have some scaling, so have to do addition before bn
        output = self.bn1(self.relu(self.deconv1(x5)) + x4)
        output = self.upsample2(output)
        output = self.upsample3(output)
        output = self.upsample4(output)
        output = self.upsample5(output)
        output = self.classifier(output)
        return output


class FCN8s(FCN):
    def __init__(self, n_class: int = 2):
        super().__init__(n_class)
        del self.upsample1, self.upsample2
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        nn.init.kaiming_normal_(self.deconv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.deconv1.bias)
        nn.init.kaiming_normal_(self.deconv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.deconv2.bias)

    def forward(self, x):
        x1 = self.maxpooling_block1(x)
        x2 = self.maxpooling_block2(x1)
        x3 = self.maxpooling_block3(x2)
        x4 = self.maxpooling_block4(x3)
        x5 = self.maxpooling_block5(x4)
        # we have to ensure both part have some scaling, so have to do addition before bn
        output = self.bn1(self.relu(self.deconv1(x5)) + x4)
        output = self.bn2(self.relu(self.deconv2(output)) + x3)
        output = self.upsample3(output)
        output = self.upsample4(output)
        output = self.upsample5(output)
        output = self.classifier(output)
        return output


class _FCNConvBlock(nn.Module):
    """ block for down-sampling in FCN (VGG), contains conv + bn + relu """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sequence(x)


class _FCNDeconvBlock(nn.Module):
    """ block for up-sampling in FCN, contains trans-conv + relu + bn """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, dilation=1,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.sequence(x)
