from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# try:
#     from encoding.nn import SyncBatchNorm

#     _BATCH_NORM = SyncBatchNorm
# except:
#     _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

_BATCH_NORM = nn.BatchNorm2d

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, stride, 0, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch, eps=1e-5, momentum=0.999)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, 1, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch, eps=1e-5, momentum=0.999)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999)
        self.relu3 = nn.ReLU()
        if downsample:
            self.downsample = nn.Sequential()
            self.downsample.add_module("0", nn.Conv2d(in_ch, out_ch, 1, stride, 0, 1, bias=False))
            self.downsample.add_module("1", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))
            self.downsample.add_module("2", nn.ReLU())
        else:
            self.downsample = lambda x: x
        # self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        # self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        # self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        # self.shortcut = (
        #     _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
        #     if downsample
        #     else lambda x: x  # identity
        # )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h += self.downsample(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "{}".format(i),
                # "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, in_ch, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class ResNet_Dilated(nn.Module):

    def __init__(self, in_ch, multi_grids=[1, 2, 4], output_stride=16, backbone='resnet101'):
        super(ResNet_Dilated, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        else:
            raise ValueError("wrong value for output_stride, which should be 8 or 16")

        ch = [64 * 2 ** p for p in range(6)]

        self.conv1 = nn.Conv2d(in_ch, ch[0], 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(ch[0], eps=1e-5, momentum=0.999)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)

        if backbone == 'resnet50':
            n_blocks=[3, 4, 6, 3]
            self.layer1 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
            self.layer2 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
            self.layer3 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
            self.layer4 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
            self.filters = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            n_blocks=[3, 4, 23, 3]
            self.layer1 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
            self.layer2 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
            self.layer3 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
            self.layer4 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
            self.filters = [256, 512, 1024, 2048]
        else:
            raise ValueError("Invalid indicator for backbone, which should be selected from {'resnet50', 'resnet101'}")
    
    def init_weights(self, pretrained=''):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda': 'cpu'})
            model_dict = self.state_dict()
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias') 
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))
        
    def get_filters(self):
        return self.filters
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool1(h)
        outs = []
        h = self.layer1(h)
        outs.append(h)
        h = self.layer2(h)
        outs.append(h)
        h = self.layer3(h)
        outs.append(h)
        h = self.layer4(h)
        outs.append(h)
        return outs

if __name__ == "__main__":
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = ResNet_Dilated(3)
    net.init_weights('/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth')
    # pretrained_dict = torch.load('/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth')
    # for k, v in net.state_dict().items():
    # for k, v in pretrained_dict.items():
        # print(k)
    print(net(input)[0].size())
    print(net(input)[1].size())
    print(net(input)[2].size())
    print(net(input)[3].size())