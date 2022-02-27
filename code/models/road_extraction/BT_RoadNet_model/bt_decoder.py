import torch
from torch import nn
import torch.nn.functional as F
from .bt_backbone import BasicBlock, Bottleneck

class Upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Upsample_layer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, size, stride=size, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class BT_Decoder(nn.Module):
    def __init__(self, block, layers, inplanes, side_out_ch):
        super(BT_Decoder, self).__init__()

        self.inplanes = inplanes
        self.layer1 = self._make_layer(block, 1024, 512, 512, layers[0], stride=1)
        self.upsample1 = Upsample_layer(512, 512, 2)
        self.layer2 = self._make_layer(block, 1024, 512, 256, layers[1], stride=1)
        self.upsample2 = Upsample_layer(256, 256, 2)
        self.layer3 = self._make_layer(block, 512, 256, 128, layers[2], stride=1)
        self.upsample3 = Upsample_layer(128, 128, 2)
        self.layer4 = self._make_layer(block, 256, 128, 64, layers[3], stride=1)
        self.upsample4 = Upsample_layer(64, 64, 2)
        self.layer5 = self._make_layer(block, 128, 64, 64, layers[4], stride=1)

        # self.side6 = nn.Conv2d(512, side_out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, side_out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, side_out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, side_out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, side_out_ch, 3, padding=1)
        self.side1 = nn.Conv2d(64, side_out_ch, 3, padding=1)

        # Initalization
        # tensorflow source code: slim.variance_scaling_initializer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 此处类似 resnet论文中的torch.nn.init.kaiming_uniform(), 但又不一样
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, midplanes, outplanes, blocks, stride=1):
        downsample_1 = None
        downsample_2 = None

        if stride != 1 or inplanes != midplanes:
            downsample_1 = nn.Sequential(
                nn.Conv2d(inplanes, midplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(midplanes),
            )
        
        if stride != 1 or midplanes != outplanes:
            downsample_2 = nn.Sequential(
                nn.Conv2d(midplanes, outplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(block(inplanes, midplanes, stride, downsample_1))
        for i in range(1, blocks - 1):
            layers.append(block(midplanes, midplanes))
        layers.append(block(midplanes, outplanes, stride, downsample_2))
        return nn.Sequential(*layers)

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
    
    def forward(self, features):
        assert len(features) == 5
        s5 = self.layer1(features[4])
        s4 = self.layer2(torch.cat([self.adjust_size(self.upsample1(s5), features[3]), features[3]], dim=1))
        s3 = self.layer3(torch.cat([self.adjust_size(self.upsample2(s4), features[2]), features[2]], dim=1))
        s2 = self.layer4(torch.cat([self.adjust_size(self.upsample3(s3), features[1]), features[1]], dim=1))
        s1 = self.layer5(torch.cat([self.adjust_size(self.upsample4(s2), features[0]), features[0]], dim=1))

        x_h, x_w = s1.size(2), s1.size(3)

        d5 = F.interpolate(self.side5(s5), size=(x_h, x_w), mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.side4(s4), size=(x_h, x_w), mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.side3(s3), size=(x_h, x_w), mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.side2(s2), size=(x_h, x_w), mode='bilinear', align_corners=True)
        d1 = self.side1(s1)

        return s1, d5, d4, d3, d2, d1

def make_decoder(side_out_ch):
    return BT_Decoder(BasicBlock, [3, 3, 3, 3, 3], 512 * 2, side_out_ch)

if __name__ == '__main__':
    m = make_decoder()
    num_params = 0
    for param in m.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    f5 = torch.autograd.Variable(torch.randn(8, 1024, 16, 16))
    f4 = torch.autograd.Variable(torch.randn(8, 512, 32, 32))
    f3 = torch.autograd.Variable(torch.randn(8, 256, 64, 64))
    f2 = torch.autograd.Variable(torch.randn(8, 128, 128, 128))
    f1 = torch.autograd.Variable(torch.randn(8, 64, 256, 256))
    out = m([f1, f2, f3, f4, f5])
    for o in out:
        print(o.size())