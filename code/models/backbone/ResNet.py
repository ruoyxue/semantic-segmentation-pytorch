import torch
from torch import nn
from torchvision import models
from torch.utils import model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_dirs = {
    'resnet18': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet18-5c106cde.pth',
    'resnet34': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet34-333f7ec4.pth',
    'resnet50': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet50-19c8e357.pth',
    'resnet101': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth',
    'resnet152': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True):
    model = models.resnet18(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(model_dirs['resnet18']))
    return model

def resnet34(pretrained=True):
    model = models.resnet34(pretrained=False)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_dirs['resnet34'], model_dir=model_dirs['resnet34']))
        model.load_state_dict(torch.load(model_dirs['resnet34']))
    return model

def resnet50(pretrained=True):
    model = models.resnet50(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(model_dirs['resnet50']))
    return model

def resnet101(pretrained=True):
    model = models.resnet101(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(model_dirs['resnet101']))
    return model

def resnet152(pretrained=True):
    model = models.resnet152(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(model_dirs['resnet152']))
    return model

class ResNet(nn.Module):
    def __init__(self, in_ch, backbone='resnet101', pretrained_flag=False):
        super(ResNet, self).__init__()
        if backbone == 'resnet18':
            # resnet = models.resnet18(pretrained=pretrained_flag)
            resnet = resnet18(pretrained=pretrained_flag)
            self.filters = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            # resnet = models.resnet34(pretrained=pretrained_flag)
            resnet = resnet34(pretrained=pretrained_flag)
            self.filters = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            # resnet = models.resnet50(pretrained=pretrained_flag)
            resnet = resnet50(pretrained=pretrained_flag)
            self.filters = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            # resnet = models.resnet101(pretrained=pretrained_flag)
            resnet = resnet101(pretrained=pretrained_flag)
            self.filters = [256, 512, 1024, 2048]
        else:
            raise ValueError("Invalid indicator for backbone, which should be selected from {'resnet18', 'resnet34', 'resnet50', 'resnet101'}")
        
        self.firstconv = nn.Conv2d(in_ch, 64, 7, 2, 3)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(3, 2, 1)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
    
    def get_filters(self):
        return self.filters
    
    def forward(self, x):
        out = []
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        out.append(e1)
        e2 = self.encoder2(e1)
        out.append(e2)
        e3 = self.encoder3(e2)
        out.append(e3)
        e4 = self.encoder4(e3)
        out.append(e4)
        return out


if __name__ == "__main__":
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = ResNet(3, backbone='resnet101')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    print(net(input)[0].size())
    print(net(input)[1].size())
    print(net(input)[2].size())
    print(net(input)[3].size())

