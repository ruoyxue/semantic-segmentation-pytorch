import torch
from torch import nn

class Spatial_Awareness_Module(nn.Module):
    def __init__(self, in_ch, r=16):
        super(Spatial_Awareness_Module, self).__init__()
        assert in_ch % r == 0
        self.conv1 = nn.Conv2d(in_ch, in_ch // r, 1)
        self.norm1 = nn.BatchNorm2d(in_ch // r)

        self.conv2 = nn.Conv2d(in_ch // r, in_ch // r, 3, dilation=4, padding=4, bias=False)
        self.norm2 = nn.BatchNorm2d(in_ch // r)
        self.conv3 = nn.Conv2d(in_ch // r, in_ch // r, 3, dilation=4, padding=4, bias=False)
        self.norm3 = nn.BatchNorm2d(in_ch // r)

        self.conv4 = nn.Conv2d(in_ch // r, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x

class Channel_Awareness_Module(nn.Module):
    def __init__(self, in_ch, r=16):
        super(Channel_Awareness_Module, self).__init__()
        assert in_ch % r == 0
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_ch, in_ch // r, 1, bias=True)
        self.norm1 = nn.BatchNorm2d(in_ch // r)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch // r, in_ch, 1, bias=True)
    
    def forward(self, x):
        x = self.gap(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Global_Awareness_Module(nn.Module):
    def __init__(self, in_ch, r=16):
        super(Global_Awareness_Module, self).__init__()
        self.sam = Spatial_Awareness_Module(in_ch, r)
        self.cam = Channel_Awareness_Module(in_ch, r)
    
    def forward(self, x):
        fs = self.sam(x)
        fc = self.cam(x)
        # fs: 1*1*H*W
        # fc: 1*C*1*1
        fg = torch.sigmoid(fs * fc)
        out = x * fg + x
        return out

if __name__ == '__main__':
    model = Global_Awareness_Module(2048)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 2048, 16, 16))
    x = model(input)
    print(x.size())