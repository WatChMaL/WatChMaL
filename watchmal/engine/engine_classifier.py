import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate


class ClassifierEngine(pl.LightningModule):

    def __init__(self, network_config, train_config):
        super().__init__()

        self.network = instantiate(network_config)
        self.network = self.network.float()
        self.learning_rate = train_config.learning_rate
        self.weight_decay = train_config.weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch["data"].float()
        labels = batch["labels"].long()
        pred_labels = self(x)
        loss = F.cross_entropy(pred_labels, labels)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        x = batch["data"].float()
        labels = batch["labels"].long()
        pred_labels = self(x)
        loss = F.cross_entropy(pred_labels, labels)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys(
            {
                "val_loss": "test_loss"
            }
        )
        return result
