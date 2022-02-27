import torch
from torch import nn
import torch.nn.functional as F

nonlinearity = nn.ReLU

ALIGN_CORNERS = True

class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()
 
        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
 
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1,bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)
 
        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=ALIGN_CORNERS)
 
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)
 
        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding,bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
 
 
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNet_Decoder(nn.Module):
    def __init__(self, filters, out_ch, is_deconv=True):
        super(UNet_Decoder, self).__init__()
        assert len(filters) == 5
        self.center = DecoderBlock(in_channels=filters[4],
                                   n_filters=filters[4],
                                   kernel_size=3,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[4] + filters[3],
                                     n_filters=filters[3],
                                     kernel_size=3,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=3,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=3,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=3,
                                     is_deconv=is_deconv)
 
 
        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, out_ch, 1))
    
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
        assert len(x) == 5
        center = self.center(x[4])
 
        d4 = self.decoder4(torch.cat([self.adjust_size(center, x[3]), x[3]], 1))
        d3 = self.decoder3(torch.cat([self.adjust_size(d4, x[2]), x[2]], 1))
        d2 = self.decoder2(torch.cat([self.adjust_size(d3, x[1]), x[1]], 1))
        d1 = self.decoder1(torch.cat([self.adjust_size(d2, x[0]), x[0]], 1))
 
        f= self.finalconv(d1)
        return f