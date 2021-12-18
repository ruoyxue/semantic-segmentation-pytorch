import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet(nn.Module):
    def __init__(self, n_channel: int, n_class: int):
        super().__init__()
        self.n_class = n_class
        self.n_channel = n_channel
        self.leftconv1 = _UNetDoubleConvBlock(n_channel, 64)
        self.leftconv2 = _UNetDoubleConvBlock(64, 128)
        self.leftconv3 = _UNetDoubleConvBlock(128, 256)
        self.leftconv4 = _UNetDoubleConvBlock(256, 512)
        self.leftconv5 = _UNetDoubleConvBlock(512, 1024)
        self.rightconv1 = _UNetDoubleConvBlock(1024, 512)
        self.rightconv2 = _UNetDoubleConvBlock(512, 256)
        self.rightconv3 = _UNetDoubleConvBlock(256, 128)
        self.rightconv4 = _UNetDoubleConvBlock(128, 64)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)
        self._initialisation()

    def _initialisation(self):
        """ weight and bias initialisation """
        def init_weight(layer):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.leftconv1.apply(init_weight)
        self.leftconv2.apply(init_weight)
        self.leftconv3.apply(init_weight)
        self.leftconv4.apply(init_weight)
        self.leftconv5.apply(init_weight)
        self.rightconv1.apply(init_weight)
        self.rightconv2.apply(init_weight)
        self.rightconv3.apply(init_weight)
        self.rightconv4.apply(init_weight)
        self.deconv1.apply(init_weight)
        self.deconv2.apply(init_weight)
        self.deconv3.apply(init_weight)
        self.deconv4.apply(init_weight)
        self.classifier.apply(init_weight)

    def forward(self, x):
        x1 = self.leftconv1(x)
        x2 = self.leftconv2(self.maxpooling(x1))
        x3 = self.leftconv3(self.maxpooling(x2))
        x4 = self.leftconv4(self.maxpooling(x3))
        y0 = self.leftconv5(self.maxpooling(x4))
        y1 = self.rightconv1(torch.concat([self.deconv1(y0), x4], dim=1))
        y2 = self.rightconv2(torch.concat([self.deconv2(y1), x3], dim=1))
        y3 = self.rightconv3(torch.concat([self.deconv3(y2), x2], dim=1))
        y4 = self.rightconv4(torch.concat([self.deconv4(y3), x1], dim=1))
        return self.classifier(y4)


class _UNetDoubleConvBlock(nn.Module):
    """ block for down-sampling in UNet, contains two conv + bn + relu """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sequence(x)
