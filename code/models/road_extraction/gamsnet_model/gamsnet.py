import torch
import torch.nn as nn
from .ms_backbone import GAMS_backbone
from .link_decoder import LinkNet_Decoder

class GAMSNet(nn.Module):
    def __init__(self, out_ch):
        super(GAMSNet, self).__init__()
        self.backbone = GAMS_backbone([3, 4, 6, 3])
        filters = [256, 512, 1024, 2048]
        self.decoder = LinkNet_Decoder(filters, out_ch)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.decoder(features)
        return out

if __name__ == '__main__':
    model = GAMSNet(17)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 3, 512, 512))
    x = model(input)
    print(x.size())