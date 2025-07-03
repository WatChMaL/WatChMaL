import torch

from watchmal.engine.reconstruction import ReconstructionEngine


class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, target_key, model, rank, device, dump_path, label_set=None):
        """
        Parameters
        ==========
        target_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        device : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        """
        # create the directory for saving the log and dump files
        super().__init__(target_key, model, rank, device, dump_path)
        self.softmax = torch.nn.Softmax(dim=1)
        self.label_set = label_set

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        data_config
            Hydra config specifying dataset.
        loaders_config
            Hydra config specifying a list of dataloaders.
        is_distributed : bool
            Whether running in multiprocessing mode.
        seed : int
            Random seed to use to initialize dataloaders.
        """
        super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
        if self.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.label_set, self.target_key)

    def process_data(self, data):
        """Extract the event data and target from the input data dict"""
        self.data = data['data'].to(self.device)
        self.target = data[self.target_key].to(self.device)

    def forward_pass(self):
        """Compute softmax predictions for a batch of data."""
        self.model_out = self.model(self.data)
        softmax = self.softmax(self.model_out)
        outputs = {self.target_key: self.target,
                   'softmax': softmax}
        return outputs

    def compute_metrics(self):
        """Compute loss and accuracy"""
        self.loss = self.criterion(self.model_out, self.target)
        predicted_labels = torch.argmax(self.model_out, dim=-1)
        accuracy = (predicted_labels == self.target).mean()
        metrics = {'loss': self.loss,
                   'accuracy': accuracy}
        return metrics
