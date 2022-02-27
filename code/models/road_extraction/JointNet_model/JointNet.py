import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Dense_Atrous_Convs_Block

class JointNet(nn.Module):
    def __init__(self, in_ch, out_ch, growth_rates=[32, 64, 128, 256], group=8):
        super(JointNet, self).__init__()
        self.dacb1 = Dense_Atrous_Convs_Block(in_ch, growth_rates[0], group=group)
        self.dacb2 = Dense_Atrous_Convs_Block(growth_rates[0], growth_rates[1], group=group)
        self.dacb3 = Dense_Atrous_Convs_Block(growth_rates[1], growth_rates[2], group=group)

        self.net_bridge = nn.Sequential(
            nn.Conv2d(growth_rates[2], growth_rates[3], 1),
            nn.ReLU(inplace=True)
        )

        self.dc1 = nn.Sequential(
            nn.Conv2d(growth_rates[2] * 4 + growth_rates[3], growth_rates[2], 1),
            nn.ReLU(inplace=True)
        )
        self.dc2 = nn.Sequential(
            nn.Conv2d(growth_rates[1] * 4 + growth_rates[2], growth_rates[1], 1),
            nn.ReLU(inplace=True)
        )
        self.dacb_f = Dense_Atrous_Convs_Block(growth_rates[0] * 4 + growth_rates[1], growth_rates[0], group=group, downsample=False)
        self.classifier = nn.Conv2d(growth_rates[0] * 4, out_ch, 1)
    
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

    def forward(self, input):
        e1, e1_d = self.dacb1(input)
        e2, e2_d = self.dacb2(e1_d)
        e3, e3_d = self.dacb3(e2_d)
        e4 = self.net_bridge(e3_d)
        d4 = F.interpolate(e4, size=(e3.size(-2), e3.size(-1)), mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.dc1(torch.cat([d4, e3], dim=1)), size=(e2.size(-2), e2.size(-1)), mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.dc2(torch.cat([d3, e2], dim=1)), size=(e1.size(-2), e1.size(-1)), mode='bilinear', align_corners=True)
        d1 = self.dacb_f(torch.cat([d2, e1], dim=1))
        out = self.classifier(d1)
        return out

if __name__ == '__main__':
    model = JointNet(3, 1).cuda()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(2, 3, 1024, 1024)).cuda()
    x = model(input)
    print(x.size())