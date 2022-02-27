import torch
from torch import nn
from .ga_module import Global_Awareness_Module

class MS_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MS_Bottleneck, self).__init__()
        assert planes % 4 == 0
        self.expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2_1 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes // 4)
        self.conv2_2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(planes // 4)
        self.conv2_3 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(planes // 4)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.stride == 1:
            self.pooling = None
        else:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=self.stride, padding=1)

    def forward(self, x):
        residual = x

        x_ = self.conv1(x)
        x_ = self.bn1(x_)
        x_ = self.relu(x_)

        C = x_.size(1)

        x1 = x_[:, 0 : C // 4, :, :]
        x2 = x_[:, C // 4 : C * 2 // 4, :, :]
        x3 = x_[:, C * 2 // 4 : C * 3 // 4, :, :]
        x4 = x_[:, C * 3 // 4 : C, :, :]

        y1 = x1
        y2 = self.relu(self.bn2_1(self.conv2_1(x2)))
        y3 = self.relu(self.bn2_2(self.conv2_2(x3 + y2)))
        y4 = self.relu(self.bn2_3(self.conv2_3(x4 + y3)))
        y = torch.cat([y1, y2, y3, y4], dim=1)

        out = self.conv3(y)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.stride != 1:
            out = self.pooling(out)

        out += residual
        out = self.relu(out)

        return out

class GAMS_backbone(nn.Module):

    def __init__(self, layers):
        self.inplanes = 64
        super(GAMS_backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # s/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(MS_Bottleneck, 64, layers[0])
        self.ga1 = Global_Awareness_Module(256)
        self.layer2 = self._make_layer(MS_Bottleneck, 128, layers[1], stride=2)
        self.ga2 = Global_Awareness_Module(512)
        self.layer3 = self._make_layer(MS_Bottleneck, 256, layers[2], stride=2)
        self.ga3 = Global_Awareness_Module(1024)
        self.layer4 = self._make_layer(MS_Bottleneck, 512, layers[3], stride=2)

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
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = []
        x = self.layer1(x)
        out.append(x)
        x = self.ga1(x)
        x = self.layer2(x)
        out.append(x)
        x = self.ga2(x)
        x = self.layer3(x)
        out.append(x)
        x = self.ga3(x)
        x = self.layer4(x)
        out.append(x)

        return out

if __name__ == '__main__':
    model = GAMS_backbone([3, 4, 6, 3])
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(8, 3, 256, 256))
    x = model(input)
    for o in x:
        print(o.size())