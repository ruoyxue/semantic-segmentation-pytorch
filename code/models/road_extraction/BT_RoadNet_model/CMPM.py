import torch
from torch import nn
from torch.nn import functional as F
from .bt_backbone import make_backbone
from .bridge import Bridge
from .bt_decoder import make_decoder

class CMPM(nn.Module):
    def __init__(self, in_ch, ms_ks, side_out_ch):
        super(CMPM, self).__init__()
        self.backbone = make_backbone(in_ch)
        self.bridge = Bridge(512, ms_ks)
        self.side6 = nn.Conv2d(512, side_out_ch, 3, padding=1)
        self.decoder = make_decoder(side_out_ch)
    
    def forward(self, x):
        out1, out2 = self.backbone(x) # out1 is for decoder, and out2 is for FMPM
        s6 = self.bridge(out1[-1])
        out1[-1] = torch.cat([s6, out1[-1]], dim=1)
        s1, d5, d4, d3, d2, d1 = self.decoder(out1)
        out2.append(s1)
        d6 = F.interpolate(self.side6(s6), scale_factor=16, mode='bilinear', align_corners=True)

        c = torch.cat(out2, dim=1) # coarse map
        d = [d6, d5, d4, d3, d2, d1] # side output

        return c, d

if __name__ == '__main__':
    cmpm = CMPM(3, 5, 2)
    num_params = 0
    for param in cmpm.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 3, 256, 256))
    c, d = cmpm(input)
    print(c.size())
    for o in d:
        print(o.size())