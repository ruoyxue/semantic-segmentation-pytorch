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