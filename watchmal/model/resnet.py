import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


class ShiftInvariantConv2d(nn.Conv2d):
    """
    Conv2d whose output is invariant to an arbitrary constant added to a subset of input channels.
    The constant can vary per example (event), but not per channel, and has no effect on the output of the convolution.
    This means that the convolution only depends on relative differences between values within the receptive field for
    those channels, and the output is independent of their absolute scale.

    This is achieved by constraining the weights to sum to zero over spatial dimensions and the specified input channels
    by subtracting the mean of the respective weights that multiply any input value of the specified channels.

    Parameters
    ----------
    constrained_in_channels : list[int]
        Input channel indices to be constrained to be shift invariant.
    All other parameters are identical to nn.Conv2d.
    """

    def __init__(self, constrained_in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_cidx", torch.tensor(constrained_in_channels, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.clone()
        w[:, self._cidx] -= w[:, self._cidx].mean(dim=(1, 2, 3), keepdim=True)
        return self._conv_forward(x, w, self.bias)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_pad_mode='zeros', norm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride, conv_pad_mode)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding_mode=conv_pad_mode)
        self.bn2 = norm(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_pad_mode='zeros', norm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding_mode=conv_pad_mode)
        self.bn2 = norm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)
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

    def __init__(self, block, layers, num_input_channels, num_output_channels, zero_init_residual=False,
                 first_kernel_size=1, first_stride=1, shift_inv_channels=None, conv_pad_mode='zeros',
                 group_norm=False, n_groups=32, gradient_checkpointing=False):
        if group_norm:
            class GroupNorm(nn.GroupNorm):
                def __init__(self, num_channels):
                    super().__init__(n_groups, num_channels)
            self.norm = GroupNorm
        else:
            self.norm = nn.BatchNorm2d

        super(ResNet, self).__init__()

        self.inplanes = 64

        pad_l = first_kernel_size // 2
        pad_r = first_kernel_size - 1 - pad_l
        self.pad1 = lambda x: F.pad(x, (pad_l, pad_r, pad_l, pad_r), mode="constant" if conv_pad_mode == "zeros" else conv_pad_mode)
        if shift_inv_channels is None:
            self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=first_kernel_size, stride=first_stride, padding=0, bias=False)
        else:
            if conv_pad_mode == "zeros":
                pad1 = self.pad1
                def replace_inv_channel_pad(x):
                    values = x[:, shift_inv_channels, 0:1, 0:1]
                    x = pad1(x)
                    x[:, shift_inv_channels, :pad_l, :] = values
                    x[:, shift_inv_channels, -pad_r:, :] = values
                    x[:, shift_inv_channels, :, :pad_l] = values
                    x[:, shift_inv_channels, :, -pad_r:] = values
                    return x
                self.pad1 = replace_inv_channel_pad
            self.conv1 = ShiftInvariantConv2d(shift_inv_channels, num_input_channels, 64, kernel_size=first_kernel_size, stride=first_stride, padding=0, bias=False)
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, conv_pad_mode=conv_pad_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, conv_pad_mode=conv_pad_mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_output_channels)

        self.use_grad_checkpointing = gradient_checkpointing

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    def _make_layer(self, block, planes, blocks, stride=1, conv_pad_mode='zeros'):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, conv_pad_mode, self.norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_pad_mode=conv_pad_mode, norm=self.norm))

        return nn.Sequential(*layers)


    def _run_layer(self, layer, x):
        """Run a residual layer group, with optional gradient checkpointing."""
        if self.use_grad_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint_sequential(
                layer, len(layer), x, use_reentrant=False
            )
        return layer(x)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self._run_layer(self.layer1, x)
        x = self._run_layer(self.layer2, x)
        x = self._run_layer(self.layer3, x)
        x = self._run_layer(self.layer4, x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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
