import torch

from watchmal.engine.reconstruction import ReconstructionEngine
from collections.abc import Mapping


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, target_key, model, rank, device, dump_path, target_scale_offset=0, target_scale_factor=1):
        """
        Parameters
        ==========
        target_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        device : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        target_scale_offset : float or dict of float
            Offset to subtract from target values when calculating the loss, or dict of offsets for each target
        target_scale_factor : float or dict of float
            Scale factor to divide target values by when calculating the loss, or dict of scale factors for each target
        """
        # create the directory for saving the log and dump files
        super().__init__(target_key, model, rank, device, dump_path)
        if isinstance(self.target_key, str):
            self.target_key = [self.target_key]
        self.target_sizes = None
        if isinstance(target_scale_offset, Mapping):  # each target has its own offset
            self.offset = {t: torch.tensor(target_scale_offset.get(t, 0)).to(self.device) for t in target_key}
        else:  # each target has the same offset
            self.offset = {t: torch.tensor(target_scale_offset).to(self.device) for t in target_key}
        if isinstance(target_scale_factor, Mapping):  # each target has its own scale
            self.scale = {t: torch.tensor(target_scale_factor.get(t, 1)).to(self.device) for t in target_key}
        else:  # each target has the same scale
            self.scale = {t: torch.tensor(target_scale_factor).to(self.device) for t in target_key}

    def process_data(self, data):
        """Extract the event data and target from the input data dict"""
        self.data = data['data'].to(self.device)
        self.target = {t: data[t].to(self.device) for t in self.target_key}
        # First time we get data, determine the target sizes
        if self.target_sizes is None:
            self.target_sizes = [v.shape[1] if len(v.shape) > 1 else 1 for v in self.target.values()]

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss and predicted values
        """
        with torch.set_grad_enabled(train):
            # scale and stack the targets for calculating the loss
            target = torch.column_stack([(v - self.offset[t]) / self.scale[t] for t, v in self.target.items()])
            # evaluate the model on the data and reshape output to match the target
            model_out = self.model(self.data).reshape(target.shape)
            # calculate the loss
            self.loss = self.criterion(model_out, target)
            # split the output for each target
            split_model_out = torch.split(model_out, self.target_sizes, dim=1)
            # return outputs including the unscaled target dictionary plus elements for the corresponding predictions
            outputs = self.target | {"predicted_"+t: o*self.scale[t] + self.offset[t]
                                     for t, o in zip(self.target.keys(), split_model_out)}
            metrics = {'loss': self.loss}
        return outputs, metrics
