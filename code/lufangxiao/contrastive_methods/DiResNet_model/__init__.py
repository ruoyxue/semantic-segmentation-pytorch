from torchvision.models import resnet
from .DirectionNet import DirectionNet
from .DiResNet import FCN_Ref
import torch
import torch.nn as nn

def build_model(in_channels=3, num_classes=1, pretrained=True):
    return FCN_Ref(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)

def build_aux_part(in_channels=1, rwidth=7, range_detect=9, rescale=False):
    dir_net = DirectionNet(in_channels=in_channels, rwidth=rwidth, range_detect=range_detect, rescale=rescale)
    struc_loss = nn.MSELoss()
    dir_loss = nn.CrossEntropyLoss()
    return dir_net, struc_loss, dir_loss