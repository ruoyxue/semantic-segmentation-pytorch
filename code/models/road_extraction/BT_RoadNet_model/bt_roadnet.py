import torch
from torch import nn
from .CMPM import CMPM
from .FMPM import FMPM

class BT_RoadNet(nn.Module):
    def __init__(self, in_ch, k, out_ch):
        super(BT_RoadNet, self).__init__()
        self.cmpm = CMPM(in_ch, k, out_ch)
        self.fmpm = FMPM(out_ch)
    
    def forward(self, x):
        coarse_map, side_outputs = self.cmpm(x)
        d0 = self.fmpm(coarse_map)
        side_outputs.append(d0)

        side_outputs = side_outputs[::-1] # outputs is d0, d1, d2, d3, d4, d5, d6
        return side_outputs

if __name__ == '__main__':
    m = BT_RoadNet(3, 5, 1)
    num_params = 0
    for param in m.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(1, 3, 1500, 1500))
    out = m(input)
    for o in out:
        print(o.size())