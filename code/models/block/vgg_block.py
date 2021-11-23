"""
Pytorch implementation of VGG blocks

Reference: https://github.com/chengyangfu/pytorch-vgg-cifar10
"""

import torch.nn as nn
import torch
from typing import List, Union, Tuple


class VGGBlock(nn.Module):
    """ Base class for VGG blocks """
    def __init__(self, config, bn_flag):
        super().__init__()
        self.sequence = VGGBlock.pile_layers(config, bn_flag)
        self._initialisation()

    def _initialisation(self):
        """ initialise weight and bias of sequence """
        def init_weight(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.sequence.apply(init_weight)

    @staticmethod
    def pile_layers(config: List[Union[Tuple[int, int], str]], bn_flag: bool = False):
        """
        :param config: config for layers
        :param bn_flag: True to use batch normalisation before relu
        :return: nn.Sequential
        """
        layers = []
        for element in config:
            if element == "maxpooling":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if isinstance(element, tuple):
                layers.append(nn.Conv2d(element[0], element[1], kernel_size=3, padding=1))
                if bn_flag:
                    layers.append(nn.BatchNorm2d(element[1]))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)


class VGG11(VGGBlock):
    def __init__(self):
        config = [(3, 64), "maxpooling", (64, 128), "maxpooling", (128, 256), (256, 256), "maxpooling",
                  (256, 512), (512, 512), "maxpooling", (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=False)


class VGG11withBN(VGGBlock):
    def __init__(self):
        config = [(3, 64), "maxpooling", (64, 128), "maxpooling", (128, 256), (256, 256), "maxpooling",
                  (256, 512), (512, 512), "maxpooling", (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=True)


class VGG13(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  "maxpooling", (256, 512), (512, 512), "maxpooling", (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=False)


class VGG13withBN(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  "maxpooling", (256, 512), (512, 512), "maxpooling", (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=True)


class VGG16(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  (256, 256), "maxpooling", (256, 512), (512, 512), (512, 512), "maxpooling", (512, 512), (512, 512),
                  (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=False)


class VGG16withBN(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  (256, 256), "maxpooling", (256, 512), (512, 512), (512, 512), "maxpooling", (512, 512), (512, 512),
                  (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=True)


class VGG19(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  (256, 256), (256, 256), "maxpooling", (256, 512), (512, 512), (512, 512), (512, 512),
                  "maxpooling", (512, 512), (512, 512), (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=False)


class VGG19withBN(VGGBlock):
    def __init__(self):
        config = [(3, 64), (64, 64), "maxpooling", (64, 128), (128, 128), "maxpooling", (128, 256), (256, 256),
                  (256, 256), (256, 256), "maxpooling", (256, 512), (512, 512), (512, 512), (512, 512),
                  "maxpooling", (512, 512), (512, 512), (512, 512), (512, 512), "maxpooling"]
        super().__init__(config=config, bn_flag=True)

