import torch.nn as nn
from hydra.utils import instantiate


class Classifier(nn.Module):

    def __init__(self, feature_extractor, classification_network, num_classes):
        super().__init__()

        # Classifier fully connected layers
        self.feature_extractor = instantiate(feature_extractor)
        classification_kwargs = {
            "num_classes": num_classes,
            "num_inputs": feature_extractor.num_output_channels
        }
        self.classification_network = instantiate(classification_network, **classification_kwargs)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classification_network(x)
        return x


class ResNetFullyConnected(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.cl_fc1 = nn.Linear(num_inputs, int(num_inputs // 2))
        self.cl_fc2 = nn.Linear(int(num_inputs // 2), int(num_inputs // 4))
        self.cl_fc3 = nn.Linear(int(num_inputs // 4), int(num_inputs // 8))
        self.cl_fc4 = nn.Linear(int(num_inputs // 8), num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.cl_fc1(x))
        x = self.relu(self.cl_fc2(x))
        x = self.relu(self.cl_fc3(x))
        x = self.cl_fc4(x)
        return x


class PointNetFullyConnected(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        min_channels = 256
        channels_1 = max(num_inputs // 2, min_channels)
        channels_2 = max(num_inputs // 4, min_channels)
        self.fc1 = nn.Linear(num_inputs, channels_1)
        self.fc2 = nn.Linear(channels_1, channels_2)
        self.fc3 = nn.Linear(channels_2, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(channels_1)
        self.bn2 = nn.BatchNorm1d(channels_2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x