"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn

from code.models.backbone import SwinTransformer
from code.models.decoder.LinkNet_Decoder import LinkNet_Decoder

class Swin_LinkNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=None, backbone='swin-t'):
        super(Swin_LinkNet, self).__init__()

        self.backbone = SwinTransformer(in_chans=num_channels, pretrain_img_size=512, window_size=8, backbone=backbone)
        if pretrained is not None:
            self.backbone.init_weights(pretrained)

        filters = self.backbone.get_filters()
        self.decoder = LinkNet_Decoder(filters, num_classes)
        self.name = "LinkNet_Swin"

    def forward(self, input):
        # Encoder
        x = self.backbone(input)

        # Decoder
        out = self.decoder(x)

        return out

if __name__ == "__main__":
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = Swin_LinkNet(num_classes=17, num_channels=3)
    print(net(input).size())