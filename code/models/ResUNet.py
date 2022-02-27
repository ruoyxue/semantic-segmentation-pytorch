import torch
from torch import nn
from code.models.backbone.ResNet import ResNet
from code.models.decoder.UNet_Decoder import UNet_Decoder



# ResNet类
class ResUNet(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='resnet101', pretrained_flag=True):
        super(ResUNet, self).__init__()

        resnet = ResNet(in_ch, backbone=backbone, pretrained_flag=pretrained_flag)

        assert backbone in ['resnet18', 'resnet34', 'resnet50','resnet101']
        filters = resnet.get_filters()
        filters.insert(0, 64)

        self.firstconv = resnet.firstconv
        self.firstbn = resnet.firstbn
        self.firstrelu = resnet.firstrelu
        self.firstmaxpool = resnet.firstmaxpool
        self.encoder1 = resnet.encoder1
        self.encoder2 = resnet.encoder2
        self.encoder3 = resnet.encoder3
        self.encoder4 = resnet.encoder4

        self.decoder = UNet_Decoder(filters, n_classes)

        self.name = "ResUNet"



    def forward(self, x):
        # Encoder
        enc_out = []
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        enc_out.append(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        enc_out.append(e1)
        e2 = self.encoder2(e1)
        enc_out.append(e2)
        e3 = self.encoder3(e2)
        enc_out.append(e3)
        e4 = self.encoder4(e3)
        enc_out.append(e4)

        # Decoder
        out = self.decoder(enc_out)

        return out


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 1500, 1500)).to(device)
    net = ResUNet(3, 17).to(device)
    print(net(input).size())