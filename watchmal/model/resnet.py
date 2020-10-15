import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2x2(in_planes, out_planes, stride=1):
    """2x2 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv4x4(in_planes, out_planes, stride=1):
    """4x4 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        if downsample is None:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            if planes < 512:
                self.conv1 = conv4x4(inplanes, planes, stride=2)
            else:
                self.conv1 = conv2x2(inplanes, planes)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        if downsample is None:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            if planes < 512:
                self.conv2 = conv4x4(planes, planes, stride=2)
            else:
                self.conv2 = conv2x2(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_input_channels, num_output_channels, zero_init_residual=False):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.layer0 = BasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.unroll_size = 128 * block.expansion
        self.bool_deep = False

        self.conv3a = nn.Conv2d(self.unroll_size, self.unroll_size, kernel_size=(4, 4), stride=(1, 1))
        self.conv3b = nn.Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1, 4), stride=(1, 1))
        self.conv3c = nn.Conv2d(self.unroll_size, self.unroll_size, kernel_size=(2, 2), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(self.unroll_size)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                self.fc1 = nn.Linear(self.unroll_size, int(self.unroll_size / 2))
                self.fc2 = nn.Linear(int(self.unroll_size / 2), num_output_channels)
                self.bool_deep = True
                break

        if not self.bool_deep:
            self.fc1 = nn.Linear(self.unroll_size, num_output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 512:
                downsample = nn.Sequential(
                    conv4x4(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv2x2(self.inplanes, planes * block.expansion),
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
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if x.size()[-2:] == (4, 4):
            x = self.conv3a(x)
        elif x.size()[-2:] == (1, 1):
            x = self.conv3b(x)
        elif x.size()[-2:] == (2, 2):
            x = self.conv3c(x)
        
        x = self.bn3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        if self.bool_deep:
            x = self.relu(self.fc2(x))
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model feature extractor.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model feature extractor.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model feature extractor.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model feature extractor.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model feature extractor.
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
