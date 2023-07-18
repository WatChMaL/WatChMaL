import torch.nn as nn
from hydra.utils import instantiate

class Regression(nn.Module):
    def __init__(self, feature_extractor, regression_alg):
        super().__init__()

        # Classifier fully connected layers
        self.feature_extractor = instantiate(feature_extractor)
        regression_kwargs = {
            "num_inputs": feature_extractor.num_output_channels
        }
        self.regression_network = instantiate(regression_alg, **regression_kwargs)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regression_network(x)
        return x

class LinearRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.cl_fc1 = nn.Linear(num_inputs, int(num_inputs // 2))
        self.cl_fc2 = nn.Linear(int(num_inputs // 2), int(num_inputs // 4))
        self.cl_fc3 = nn.Linear(int(num_inputs // 4), int(num_inputs // 8))
        self.cl_fc4 = nn.Linear(int(num_inputs // 8), num_outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.cl_fc1(x))
        x = self.relu(self.cl_fc2(x))
        x = self.relu(self.cl_fc3(x))
        x = self.cl_fc4(x)
        return x

class PassThrough(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
    def forward(self, x):
        return x