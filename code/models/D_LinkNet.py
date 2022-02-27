"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn

from code.models.backbone.ResNet import ResNet
from code.lufangxiao.module.DBlock import Dblock
from code.models.decoder.LinkNet_Decoder import LinkNet_Decoder

class DLinkNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, backbone='resnet50', pretrained_flag=True):
        super(DLinkNet, self).__init__()

        assert backbone in ['resnet18', 'resnet34', 'resnet50','resnet101']
        
        self.backbone = ResNet(num_channels, backbone, pretrained_flag)
        filters = self.backbone.get_filters()
        self.dblock = Dblock(filters[3])
        self.decoder = LinkNet_Decoder(filters, num_classes)
        self.name = "DLinkNet_" + backbone

    def forward(self, input):
        # Encoder
        x = self.backbone(input)

        # Center
        x[3] = self.dblock(x[3])

        # Decoder
        out = self.decoder(x)

        return out

if __name__ == "__main__":
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = DLinkNet(num_classes=1, num_channels=3, backbone='resnet18', pretrained_flag=True)
    print(net(input).size())