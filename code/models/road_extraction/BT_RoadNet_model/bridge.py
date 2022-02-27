import torch
from torch import nn
from torch.nn import functional as F
import math

class basic_conv(nn.Module):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1, dilation=1):
        super(basic_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Spatial_Context_Module(nn.Module):
    def __init__(self, in_ch, ms_ks):
        super(Spatial_Context_Module, self).__init__()
        self.message_passing_1 = nn.ModuleList()
        self.message_passing_2 = nn.ModuleList()
        self.message_passing_1.add_module('up_down_1', basic_conv(in_ch, in_ch, kernel_size=(1, ms_ks), stride=1, padding=(0, ms_ks // 2)))
        self.message_passing_2.add_module('up_down_2', basic_conv(in_ch, in_ch, kernel_size=(2, ms_ks), stride=1, padding=(0, ms_ks // 2)))
        self.message_passing_1.add_module('down_up_1', basic_conv(in_ch, in_ch, kernel_size=(1, ms_ks), stride=1, padding=(0, ms_ks // 2)))
        self.message_passing_2.add_module('down_up_2', basic_conv(in_ch, in_ch, kernel_size=(2, ms_ks), stride=1, padding=(0, ms_ks // 2)))
        self.message_passing_1.add_module('left_right_1',
                                        basic_conv(in_ch, in_ch, kernel_size=(ms_ks, 1), stride=1, padding=(ms_ks // 2, 0)))
        self.message_passing_2.add_module('left_right_2',
                                        basic_conv(in_ch, in_ch, kernel_size=(ms_ks, 2), stride=1, padding=(ms_ks // 2, 0)))
        self.message_passing_1.add_module('right_left_1',
                                        basic_conv(in_ch, in_ch, kernel_size=(ms_ks, 1), stride=1, padding=(ms_ks // 2, 0)))
        self.message_passing_2.add_module('right_left_2',
                                        basic_conv(in_ch, in_ch, kernel_size=(ms_ks, 2), stride=1, padding=(ms_ks // 2, 0)))
    
    def forward(self, x):
        x = self.message_passing_forward(x)
        return x
    
    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv_1, ms_conv_2, v, r in zip(self.message_passing_1, self.message_passing_2, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv_1, ms_conv_2, v, r)
        return x

    def message_passing_once(self, x, conv_1, conv_2, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [conv_1(slices[0])]
        for i in range(1, len(slices)):
            out.append(conv_2(torch.cat([slices[i], out[i - 1]], dim=dim)))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

class Bridge(nn.Module):
    def __init__(self, in_ch, ms_ks):
        super(Bridge, self).__init__()
        self.cbr1 = basic_conv(in_ch, in_ch, 3, 1, 1, 1)
        self.cbr2 = basic_conv(in_ch, in_ch // 4, 3, 1, 1, 1)
        self.cbr3 = basic_conv(in_ch // 4, in_ch // 4, 3, 1, 1, 1)
        self.cbr4 = basic_conv(in_ch // 4, in_ch, 3, 1, 1, 1)
        self.scm = Spatial_Context_Module(in_ch // 4, ms_ks)
    
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.scm(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        return x

if __name__ == '__main__':
    model = Bridge(512, 5)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 512, 16, 16))
    x = model(input)
    print(x.size())