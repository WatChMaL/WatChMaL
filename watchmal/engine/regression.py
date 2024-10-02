import torch

from watchmal.engine.reconstruction import ReconstructionEngine

import analysis.utils.math as math


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, output_center=0, output_scale=1, eval_directory='/'):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        output_center : float
            Value to subtract from target values
        output_scale : float
            Value to divide target values by
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path)
        self.output_center = output_center
        self.output_scale = output_scale
        self.eval_directory=eval_directory

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
            #model_out = self.model(self.data)
            #Force float type
            scaled_target = self.scale_values(self.target).float()
            scaled_model_out = self.scale_values(model_out).float()
            self.loss = self.criterion(scaled_model_out, scaled_target)
            if self.dir is not None and train is False:
                longitudinal_component_pred = math.decompose_along_direction_pytorch(scaled_model_out[:,0:3]-scaled_target[:,0:3], self.dir)
                #longitudinal_component_true = math.decompose_along_direction_pytorch(scaled_target[:,0:3], self.dir)
            if False:
                print(f'center: {self.output_center}, scale: {self.output_scale}')
                print(f'Loss: {self.loss}, pred: {torch.mean(torch.abs(scaled_model_out),dim=0)}, target: {torch.mean(torch.abs(scaled_target), dim=0)}, train: {train}')
            outputs = {"predicted_"+self.truth_key: model_out}
            if False and (self.dir is not None and train is False):
                (numerical_bot_quantile, numerical_top_quantile) = torch.quantile(longitudinal_component_pred, torch.tensor([0.159,0.841]).to(self.device))
                numerical_median = torch.median(longitudinal_component_pred)
                #numerical_top_quantile = np.quantile(residuals_cut, 0.841)
                quantile = (torch.abs((numerical_median-numerical_bot_quantile))+torch.abs((numerical_median-numerical_top_quantile)))/2
                #print(f"pred: {scaled_model_out[:,0:3]}, true: {scaled_target[:,0:3]}, dir: {self.dir}, long pred: {longitudinal_component_pred}, long true: {longitudinal_component_true}")
                metrics = {'loss': self.loss, 'long_res':quantile}
            else:
                metrics = {'loss': self.loss}
        return outputs, metrics

    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
