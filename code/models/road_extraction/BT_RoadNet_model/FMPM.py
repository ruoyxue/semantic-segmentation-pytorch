import torch
from torch import nn
from torch.nn import functional as F
from .bridge import basic_conv
from .bt_decoder import Upsample_layer

class FMPM(nn.Module):
    def __init__(self, out_ch):
        super(FMPM, self).__init__()
        # encoder
        self.conv1 = basic_conv(256, 64, 3, padding=1)
        self.conv2 = basic_conv(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = basic_conv(64, 64, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv4 = basic_conv(64, 64, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv5 = basic_conv(64, 64, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        # bridge
        self.conv6 = basic_conv(64, 64, 3, padding=1)
        self.up1 = Upsample_layer(64, 64, 2)

        # decoder
        self.conv7 = basic_conv(128, 64, 3, padding=1)
        self.up2 = Upsample_layer(64, 64, 2)
        self.conv8 = basic_conv(128, 64, 3, padding=1)
        self.up3 = Upsample_layer(64, 64, 2)
        self.conv9 = basic_conv(128, 64, 3, padding=1)
        self.up4 = Upsample_layer(64, 64, 2)
        self.conv10 = basic_conv(128, 64, 3, padding=1)

        self.conv_out = nn.Conv2d(64, out_ch, 3, padding=1)

    def adjust_size(self, x, y):
        h_x, w_x = x.size(-2), x.size(-1)
        h_y, w_y = y.size(-2), y.size(-1)
        h = max(h_x, h_y)
        w = max(w_x, w_y)
        right_padding = int(w - w_x)
        bottom_padding = int(h - h_x)
        padding = (0, right_padding, 0, bottom_padding)
        x = F.pad(x, padding, "constant", 0)
        x = x[:, :, 0:h_y, 0:w_y]
        return x

    def forward(self, x):
        e1 = self.conv2(self.conv1(x))
        e2 = self.conv3(self.maxpool1(e1))
        e3 = self.conv4(self.maxpool2(e2))
        e4 = self.conv5(self.maxpool3(e3))

        b = self.up1(self.conv6(self.maxpool4(e4)))

        d4 = self.conv7(torch.cat([self.adjust_size(b, e4), e4], dim=1))
        d3 = self.conv8(torch.cat([self.adjust_size(self.up2(d4), e3), e3], dim=1))
        d2 = self.conv9(torch.cat([self.adjust_size(self.up3(d3), e2), e2], dim=1))
        d1 = self.conv10(torch.cat([self.adjust_size(self.up4(d2), e1), e1], dim=1))

        out = self.conv_out(d1)
        return out

if __name__ == '__main__':
    cmpm = FMPM(2)
    num_params = 0
    for param in cmpm.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 256, 256, 256))
    c = cmpm(input)
    print(c.size())