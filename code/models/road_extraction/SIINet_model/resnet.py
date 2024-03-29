'''
Personalize resnet, base torchvision.models.resnet.
Difference:
    1. Orderdict(name)
    2. Number of blocks
    3. Output stride
    4. Initalization
'''
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_dirs = {
    'resnet18': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet18-5c106cde.pth',
    'resnet34': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet34-333f7ec4.pth',
    'resnet50': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet50-19c8e357.pth',
    'resnet101': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet101-5d3b4d8f.pth',
    'resnet152': '/home/lufangxiao/GDANet/models/backbone/pretrained/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    ''' Residual block '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks, output_stride=16):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # s/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # s/4
        s = [1, 1, 1, 1, 1, 1]  # list of stride
        for i in range(int(math.log2(output_stride))-2):
            s[i] = 2
        self.layer1 = self._make_layer(block, 64, blocks[0], stride=1)  # s/4
        self.layer2 = self._make_layer(block, 128, blocks[1], stride=s[0])  # s/out_s
        self.layer3 = self._make_layer(block, 256, blocks[2], stride=s[1])
        self.layer4 = self._make_layer(block, 512, blocks[3], stride=s[2])
        self.layer5 = self._make_layer(block, 512, 3, stride=s[3])
        self.layer6 = self._make_layer(block, 256, 3, stride=s[4])
        self.layer7 = self._make_layer(block, 128, 3, stride=s[5])
        # TODO: How to reasonably make sure layer7 output dim is 512?

        # Initalization
        # tensorflow source code: slim.variance_scaling_initializer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 此处类似 resnet论文中的torch.nn.init.kaiming_uniform(), 但又不一样
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_weight(model, 'resnet18')
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_weight(model, 'resnet34')
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_weight(model, 'resnet50')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        load_weight(model, 'resnet101')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        load_weight(model, 'resnet152')
    return model


def load_weight(model, model_name):
    '''
    Due to this resnet is partially different from the orignal resnet,
    so need to use update to load pretrain weight.
    '''

    model_dict = model.state_dict()  # weight value dict
    # pretrained_dict = model_zoo.load_url(model_urls[model_name])
    pretrained_dict = torch.load(model_dirs[model_name])
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)  # Only update weights with the same key
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    model = resnet50(output_stride=8)
    input = torch.autograd.Variable(torch.randn(8, 3, 1024, 1024))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    print(model(input).size())