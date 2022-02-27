import torch
import torch.nn as nn
import torch.nn.functional as F

from code.models.backbone import SwinTransformer
from code.models.decoder.UperNet_Decoder import UperNet_Decoder

class Swin_UperNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=None, backbone='swin-t'):
        super(Swin_UperNet, self).__init__()

        self.backbone = SwinTransformer(in_chans=num_channels, pretrain_img_size=512, window_size=8, backbone=backbone)
        if pretrained is not None:
            self.backbone.init_weights(pretrained)

        filters = self.backbone.get_filters()
        self.decoder = UperNet_Decoder(filters, num_classes)
        self.name = "Swin_UperNet"

    def forward(self, input):
        # Encoder
        x = self.backbone(input)

        # Decoder
        out = self.decoder(x)
        out = F.interpolate(out, size=input.shape[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == "__main__":
    input = torch.autograd.Variable(torch.randn(1, 3, 1500, 1500))
    net = Swin_UperNet(num_classes=17, num_channels=3)
    net.eval()
    print(net(input).size())