'''
Timm is a popular library that provides a wide range of pre-trained vision models.
If you'd like to try some models without implementing them from scratch, you can use this TimmRegressor class.
Here is an instance of a SwinT model.
'''
import torch
import torch.nn as nn
import timm


class TimmRegressor(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224", # here is an instance of Swin Transformer
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        num_output_channels=3,
    ):
        super().__init__()

        self.timm = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_output_channels,
            img_size=img_size,
        )
        self.output_dim = num_output_channels

    def forward(self, x):
        out = self.timm(x)
        return out
