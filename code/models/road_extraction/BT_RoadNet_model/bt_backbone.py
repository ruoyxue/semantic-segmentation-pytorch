import torch
from torch import nn

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

class Res_Backbone(nn.Module):
    def __init__(self, in_ch, block, layers, mid_output=False):
        self.inplanes = in_ch
        self.mid_output = mid_output
        super(Res_Backbone, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, mid_output=self.mid_output)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

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

    def _make_layer(self, block, planes, blocks, stride=1, mid_output=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        if mid_output:
            layers = nn.ModuleList([])
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return layers
        else:
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)
    
    def forward(self, x):
        out1 = []
        out2 = []

        if self.mid_output:
            for i in range(len(self.layer1)):
                x = self.layer1[i](x)
                out2.append(x)
        else:
            x = self.layer1(x)
        out1.append(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        out1.append(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        out1.append(x)
        x = self.maxpool3(x)
        x = self.layer4(x)
        out1.append(x)
        x = self.maxpool4(x)
        x = self.layer5(x)
        out1.append(x)

        return out1, out2

def make_backbone(in_ch):
    return Res_Backbone(in_ch, BasicBlock, [3, 4, 6, 3, 3], mid_output=True)

if __name__ == '__main__':
    model = make_backbone()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 3, 1024, 1024))
    x = model(input)
    print(len(x[0]))
    for q in x[0]:
        print(q.size())
    print(len(x[1]))
    for p in x[1]:
        print(p.size())