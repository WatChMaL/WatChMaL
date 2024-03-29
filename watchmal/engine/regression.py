import torch

from watchmal.engine.reconstruction import ReconstructionEngine


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, truth_key, model, rank, device, dump_path, output_center=0, output_scale=1):
        """
        Parameters
        ==========
        truth_key : string
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
        super().__init__(truth_key, model, rank, device, dump_path)
        self.output_center = torch.tensor(output_center).to(self.device)
        self.output_scale = torch.tensor(output_scale).to(self.device)

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
            model_out = self.model(self.data).reshape(self.target.shape)
            scaled_target = (self.target - self.output_center) / self.output_scale
            self.loss = self.criterion(model_out, scaled_target)
            scaled_model_out = model_out * self.output_scale + self.output_center
            outputs = {"predicted_"+self.truth_key: scaled_model_out}
            metrics = {'loss': self.loss}
        return outputs, metrics
