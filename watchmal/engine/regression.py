import torch

from watchmal.engine.reconstruction import ReconstructionEngine
from collections.abc import Mapping

# define some useful metrics for different regression targets
metric_functions = {
    'positions':  # mean 3D position error
        lambda x, y: torch.mean(torch.linalg.vector_norm(x-y, dim=1)),
    'directions':  # mean angle between directions
        lambda x, y: torch.mean(torch.arccos(torch.clamp(torch.sum(x*y, dim=-1)
                                                         / torch.linalg.vector_norm(x, dim=-1), -1, 1))),
    'angles':  # mean angle between directions
        lambda x, y: torch.mean(torch.arccos(torch.cos(x[:, 0])*torch.cos(y[:, 0])
                                             + torch.sin(x[:, 0])*torch.sin(y[:, 0])*torch.cos(x[:, 1]-y[:, 1]))),
    'energies':  # mean fractional error
        lambda x, y: torch.mean((x - y) / y),
}


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, target_key, model=None, rank=None, device=None, dump_path=None,
                 target_scale_offset=0, target_scale_factor=1):
        """
        Parameters
        ==========
        target_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model : nn.Module
            Model that outputs predicted values for each regressed quantity
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
            self.offset = {t: torch.tensor(target_scale_offset.get(t, 0)).to(self.device) for t in self.target_key}
        else:  # each target has the same offset
            self.offset = {t: torch.tensor(target_scale_offset).to(self.device) for t in self.target_key}
        if isinstance(target_scale_factor, Mapping):  # each target has its own scale
            self.scale = {t: torch.tensor(target_scale_factor.get(t, 1)).to(self.device) for t in self.target_key}
        else:  # each target has the same scale
            self.scale = {t: torch.tensor(target_scale_factor).to(self.device) for t in self.target_key}
        self.target_dict = None
        self.stacked_target = None
        self.predictions = None

    def process_target(self, data):
        """Extract the event data and target from the input data dict"""
        self.target_dict = {t: data[t].to(self.device) for t in self.target_key}
        # First time we get data, determine the target sizes
        if self.target_sizes is None:
            self.target_sizes = [v.shape[-1] if len(v.shape) > 1 else 1 for v in self.target_dict.values()]
        # scale and stack the targets for calculating the loss
        self.stacked_target = torch.column_stack([(v - self.offset[t]) / self.scale[t] for t, v in self.target_dict.items()])

    def compute_outputs(self):
        """Compute predictions for a batch of data"""
        # split the output for each target
        split_model_out = torch.split(self.model_out, self.target_sizes, dim=1)
        self.predictions = {"predicted_" + t: o * self.scale[t] + self.offset[t]
                            for t, o in zip(self.target_key, split_model_out)}
        if self.target_dict is None:
            return self.predictions
        return self.target_dict | self.predictions

    def compute_metrics(self):
        self.loss = self.criterion(self.model_out, self.stacked_target)
        # return loss and metrics for the predictions
        metrics = {t+" error": metric_functions[t](self.predictions["predicted_"+t], v)
                   for t, v in self.target_dict.items() if t in metric_functions}
        metrics['loss'] = self.loss
        return metrics

    def save_state(self, suffix="", name=None):
        self.state_data["target_sizes"] = self.target_sizes
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
        if "target_sizes" in self.state_data:
            self.target_sizes = self.state_data["target_sizes"]
