import torch

from watchmal.engine.reconstruction import ReconstructionEngine


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, target_key, model, rank, device, dump_path, target_scaling=None):
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
        output_center : float
            Value to subtract from target values
        output_scale : float
            Value to divide target values by
        """
        # create the directory for saving the log and dump files
        super().__init__(target_key, model, rank, device, dump_path)
        self.output_center = torch.tensor(output_center).to(self.device)
        self.output_scale = torch.tensor(output_scale).to(self.device)
        if isinstance(self.target_key, str):
            self.target_key = [self.target_key]
        self.target_sizes = None

    def process_data(self, data):
        """Extract the event data and target from the input data dict"""
        self.data = data['data'].to(self.device)
        self.target = {k: data[k].to(self.device) for k in self.target_key}
        # First time we get data, determine the target sizes and set related things
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
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            stacked_targets = torch.column_stack(list(self.target.values()))
            model_out = self.model(self.data).reshape(stacked_targets.shape)
            scaled_target = (stacked_targets - self.output_center) / self.output_scale
            self.loss = self.criterion(model_out, scaled_target)
            scaled_model_out = model_out * self.output_scale + self.output_center
            # Outputs include the target dictionary plus corresponding elements for each prediction
            predictions = torch.split(scaled_model_out, self.target_sizes, dim=1)
            outputs = self.target | {"predicted_"+k: v for k, v in zip(self.target.keys(), predictions)}
            metrics = {'loss': self.loss}
        return outputs, metrics
