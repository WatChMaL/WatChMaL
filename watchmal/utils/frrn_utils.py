import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable



class conv3DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv3DBatchNorm, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm3d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv3DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv3DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm3d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs




class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, residual_channels, scale, concat_method = "Stack"):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.concat_method = concat_method

        self.residualWeight = 0.5
        self.poolingWeight = 0.5

        conv_unit = conv3DBatchNormRelu

        if(self.concat_method == "Stack"):
            self.conv1 = conv_unit(
                prev_channels + self.residual_channels, out_channels, k_size=(19,3,3), stride=1, padding=(9,1,1), bias=False
            )
        elif(self.concat_method == "Fusion"):
            self.conv1 = conv_unit(
                prev_channels, out_channels, k_size=(19,3,3), stride=1, padding=(9,1,1), bias=False
            )          
        else:
            raise Exception("Invalid concatenation method specified: " + self.concat_method)  

        
        self.conv2 = conv_unit(
            out_channels, out_channels, k_size=(19,3,3), stride=1, padding=(9,1,1), bias=False
        )

        self.conv_res = nn.Conv3d(out_channels, self.residual_channels, kernel_size=(19,1,1), stride=1, padding=(9,0,0))
        self.conv_down_res = nn.Conv3d(self.residual_channels,prev_channels,kernel_size=(19,1,1), stride=1, padding=(9,0,0))

    def forward(self, y, z):

        if(self.concat_method == "Fusion"):
            y = y * self.poolingWeight
            zp = self.conv_down_res(z)
            x = y.add(nn.MaxPool3d(self.scale, self.scale)(zp) * self.residualWeight)
        else:
            x = torch.cat([y, nn.MaxPool3d(self.scale, self.scale)(z)], dim=1)
        
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        #print("Test size:", y_prime.shape)

        scaleFactorTensor = torch.tensor(np.asarray(self.scale))
        upsample_size = torch.Size([dimShape * dimScale for dimShape, dimScale in zip(y_prime.shape[2:],scaleFactorTensor)])
        #upsample_size = (torch.tensor(np.asarray(scale_factor)) * y_prime.shape[2:]).shape
        #print("upsample:", upsample_size)
        x = F.upsample(x, size=upsample_size, mode="nearest")
        
        z_prime = z + x

        return y_prime, z_prime


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=1, padding = 0, strides=1):
        super(RU, self).__init__()

        self.conv1 = conv3DBatchNormRelu(
            channels, channels, k_size=kernel_size, stride=strides, padding=padding, bias=False
        )
        self.conv2 = conv3DBatchNorm(
            channels, channels, k_size=kernel_size, stride=strides, padding=padding, bias=False
        )

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming

