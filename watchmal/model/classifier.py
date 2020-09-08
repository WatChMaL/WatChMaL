import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate


class Classifier(nn.Module):

    def __init__(self, num_latent_dims, num_classes):
        super().__init__()

        # Classifier fully connected layers
        self.cl_fc1 = nn.Linear(num_latent_dims, int(num_latent_dims / 2))
        self.cl_fc2 = nn.Linear(int(num_latent_dims / 2), int(num_latent_dims / 4))
        self.cl_fc3 = nn.Linear(int(num_latent_dims / 4), int(num_latent_dims / 8))
        self.cl_fc4 = nn.Linear(int(num_latent_dims / 8), num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Fully-connected layers
        x = self.relu(self.cl_fc1(x))
        x = self.relu(self.cl_fc2(x))
        x = self.relu(self.cl_fc3(x))
        x = self.cl_fc4(x)

        return x


class ClassifierLightingModule(pl.LightningModule):

    def __init__(self, train_config, network_config, feature_extractor):
        super().__init__()

        self.feature_extractor = instantiate(feature_extractor)
        self.classifier = Classifier(network_config.num_latent_dims, network_config.num_classes)
        self.learning_rate = train_config.learning_rate
        self.weight_decay = train_config.weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def loss(self, output, target):
        return nn.CrossEntropyLoss(output, target)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        pred_labels = self(x)
        loss = F.cross_entropy(pred_labels, labels)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        pred_labels = self(x)
        loss = F.cross_entropy(pred_labels, labels)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def validation_epoch_end(self, outputs):
        return

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys(
            {
                "val_loss": "test_loss"
            }
        )
        return result