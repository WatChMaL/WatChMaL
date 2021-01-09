"""
dcgan.py
Class implementing a DCGAN
"""

# WatChMaL imports
from watchmal.model.GAN import GeneratorDiscriminator

# PyTorch imports
from torch.nn import Module, init
from torch import Tensor, randn, full
from torch.nn import Softmax, BCEWithLogitsLoss, BCELoss
from torch.optim import Adam

# Generic imports
import numpy as np


def weights_init(m):
    """
    Module for initializing custom weights.
    Custom weights drawn from a normal distribution with mean=0, stdev=0.02
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class GanNet(Module):
    
    def __init__(self, num_input_channels, num_latent_dims, num_classes, arch_key, arch_depth, train_all):
        Module.__init__(self)

        # Initialize models
        self.generator = getattr(GeneratorDiscriminator, "genresnet" + str(arch_depth))(num_input_channels=num_input_channels,
                                                                                        num_latent_dims=num_latent_dims)
        self.discriminator = getattr(GeneratorDiscriminator, "disresnet" + str(arch_depth))(num_input_channels=num_input_channels,
                                                                                            num_latent_dims=num_latent_dims)
        
        # set custom weights
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        if not train_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X, Z):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """

        # Generate a batch of images
        gen_imgs = self.generator(Z)

        # Run discriminator on real and generated images
        dis_genresults = self.discriminator(gen_imgs)
        dis_realresults = self.discriminator(X)
        
        return {'genresults': dis_genresults, 'realresults': dis_realresults, 'genimgs': gen_imgs}