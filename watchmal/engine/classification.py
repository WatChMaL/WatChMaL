import torch

from watchmal.engine.reconstruction import ReconstructionEngine


class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, label_set=None):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path)
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
                self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            model_out = self.model(self.data)
            softmax = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out, dim=-1)
            self.loss = self.criterion(model_out, self.target)
            accuracy = (predicted_labels == self.target).sum() / float(predicted_labels.nelement())
            outputs = {'softmax': softmax}
            metrics = {'loss': self.loss,
                       'accuracy': accuracy}
        return outputs, metrics
