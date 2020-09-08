import torch.nn as nn
from hydra.utils import instantiate


class ClassifierNetwork(nn.Module):

    def __init__(self, feature_extractor, network_config):
        super().__init__()

        # Classifier fully connected layers
        self.feature_extractor = instantiate(feature_extractor)
        self.cl_fc1 = nn.Linear(feature_extractor.num_encoder_outputs, int(feature_extractor.num_encoder_outputs / 2))
        self.cl_fc2 = nn.Linear(int(feature_extractor.num_encoder_outputs / 2), int(feature_extractor.num_encoder_outputs / 4))
        self.cl_fc3 = nn.Linear(int(feature_extractor.num_encoder_outputs / 4), int(feature_extractor.num_encoder_outputs / 8))
        self.cl_fc4 = nn.Linear(int(feature_extractor.num_encoder_outputs / 8), network_config.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature extractor
        x = self.feature_extractor(x)
        # Fully-connected layers
        x = self.relu(self.cl_fc1(x))
        x = self.relu(self.cl_fc2(x))
        x = self.relu(self.cl_fc3(x))
        x = self.cl_fc4(x)

        return x
