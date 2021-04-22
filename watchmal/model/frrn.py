import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

from watchmal.utils.frrn_utils import FRRU, RU, conv3DBatchNormRelu


"""
Model Specifications:
- A: Shallow Model - Single FRRU depth
- B: Deep Model - Multiple FRRU's - structure based on original paper implementation
- C: Double Channel Deep Model - Same as B with doubled feature channels at each step
- D: Mixed Convolution Deep Model - Alternative weighted average concatenation method

Dictionary Format:
- encoder: List of FRRU's that downsample
- decoder: List of FRRU's that upsample
- n_channels: Hyperparameter representing number of channels to be used outside of the FRRU units
    - i.e. number of channels to use before and after the encoding/decoding execution
    - In addition, it forms the basis for the number of residual channels to be used in each FRRU
    - Typically, larger values improve accuracy, but extend training time and model size
- concat_method: ("Stack", "Fusion")
    - Stack: Residual channels are stacked on top of pooled channels in each FRRU, regardless of pooled channel size
    - Fusion: Residual stream is fed through 19x1x1 convolution to match pooled stream channel dimension, then a 
                weighted average element-wise addition concatenation is carried out
        - Residual stream and pooling stream weights are specified in the constructor of the FRRU Class

FRRU Format:
- List of 3 elements: [n_blocks, channels, scale]
    1. n_blocks: Number of repeated FRRU's to apply with the given size/settings
    2. channels: Number of feature channels to use/output
    3. scale: 3 element tuple which represents the max pooling scale factor to apply to each dimension
        - i.e. The data shape is [19,29,40], so for ex. (1,2,2) would apply a 1x2x2 max pooling to the data
        - Given that the spacial geometry of each PMT is not associated by the index order, we never want to reduce
            the depth (19 channel) dimension, we want to leave it untouched
"""


frrn_specs_dic = {
    "A": {
        "encoder": [[3, 8, (1,2,2)]],
        "decoder": [],
        "n_channels": 4,
        "concat_method": "Stack"
    },
    "B": {
        "encoder": [[3, 8, (1,2,2)], [4, 16, (1,4,4)], [2, 32, (1,8,8)]],
        "decoder": [[2, 16, (1,4,4)], [2, 4, (1,2,2)]],
        "n_channels": 8,
        "concat_method": "Stack"
    },
    "C": {
        "encoder": [[3, 16, (1,2,2)], [4, 32, (1,4,4)], [2, 64, (1,8,8)]],
        "decoder": [[2, 32, (1,4,4)], [2, 8, (1,2,2)]],
        "n_channels": 8,
        "concat_method": "Stack"
    },
    "D": {
        "encoder": [[3, 8, (1,2,2)], [4, 16, (1,4,4)], [2, 32, (1,8,8)]],
        "decoder": [[2, 16, (1,4,4)], [2, 4, (1,2,2)]],
        "n_channels": 8,
        "concat_method": "Fusion"
    },
}

class frrn(nn.Module):


    def __init__(self, n_classes=4, model_type="C"):

        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type

        self.n_channels = frrn_specs_dic[self.model_type]["n_channels"]
        self.baseline_scale_factor = (1,2,2)
        self.concat_method = frrn_specs_dic[self.model_type]["concat_method"]

        self.conv1 = conv3DBatchNormRelu(1, self.n_channels, (19,3,3), 1, (9,1,1))

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(
                RU(
                    channels=self.n_channels,
                    kernel_size=(19,3,3),
                    padding = (9,1,1),
                    strides=1
                )
            )
            self.down_residual_units.append(
                RU(
                    channels=self.n_channels,
                    kernel_size=(19,3,3),
                    padding = (9,1,1),
                    strides=1
                )
            )

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv_3d = nn.Conv3d(self.n_channels, self.n_channels, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=False)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]
        
        # encoding
        prev_channels = self.n_channels
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        residual_channels = self.n_channels,
                        scale=scale,
                        concat_method = self.concat_method
                    ),
                )
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        residual_channels = self.n_channels,
                        scale=scale,
                        concat_method = self.concat_method
                    ),
                )
            prev_channels = channels
        
        self.merge_conv_3d = nn.Conv3d(
            prev_channels + self.n_channels, self.n_channels, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=False
        )

        self.classif_conv_3d = nn.Conv3d(
            self.n_channels, self.n_classes, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=True
        )

        self.activationFunction = nn.ReLU(inplace=False)

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv_3d(x)

        prev_channels = self.n_channels

        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool3d(y, stride=self.baseline_scale_factor , kernel_size=self.baseline_scale_factor , padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([dimShape * dimScale for dimShape, dimScale in zip(y.shape[2:],list(self.baseline_scale_factor))])
            y_upsampled = F.upsample(y, size=upsample_size, mode="trilinear", align_corners=True)

            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_upsampled, z)
            prev_channels = channels
        
        # merge streams
        x = torch.cat(
            [F.upsample(y, scale_factor=self.baseline_scale_factor , mode="trilinear", align_corners=True), z], dim=1
        )
        x = self.merge_conv_3d(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)
        
        # final 1x1 conv to get classification
        x = self.classif_conv_3d(x)

        #Activation function - ReLU
        x = self.activationFunction(x)
        

        return x
