import torch
from torch import nn
from hydra.utils import instantiate

from watchmal.engine.reconstruction import ReconstructionEngine


class JointClassificationRegression(ReconstructionEngine):
    """Engine for performing combined regression and classification"""
    def __init__(self, classification_engine, regression_engine, model, rank, device, dump_path, loss_weight = None):
        """
        Parameters
        ==========
        classification_engine : ClassifierEngine
            A fully instantiated classification engine (but instantiated without its own model, rank, device, dump_path)
        regression_engine : RegressionEngine
            A fully instantiated regression engine (but instantiated without its own model, rank, device, dump_path)
        model : nn.Module
            Model that outputs concatenated [regression_outputs, classification_logits]
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        device : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        loss_weight : float
            Weight L for loss calculation where the loss will be L * regression_loss + (1 - L) * classification_loss
            If `loss_weight` is None, then use uncertainty weighting instead
        """
        self.classification_engine = instantiate(classification_engine, rank=rank, device=device, dump_path=dump_path)
        self.regression_engine = instantiate(regression_engine, rank=rank, device=device, dump_path=dump_path)
        super().__init__(
            model=model,
            rank=rank,
            device=device,
            dump_path=dump_path,
            target_key=regression_engine.target_key + [classification_engine.target_key],
        )
        self.regression_size = None
        self.loss_weight = loss_weight
        if loss_weight is None:
            self.module.log_vars = nn.ParameterDict({
                "regression_log_var": nn.Parameter(torch.zeros(1)).squeeze(),
                "classification_log_var": nn.Parameter(torch.zeros(1)).squeeze(),
            }).to(self.device)


    def configure_loss(self, loss_config):
        """Configure losses for both engines."""
        self.classification_engine.configure_loss(loss_config["classification"])
        self.regression_engine.configure_loss(loss_config["regression"])

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """Configure data loaders shared for both engines, including label mapping for classification."""
        super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
        if self.classification_engine.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.classification_engine.label_set,
                                                           self.classification_engine.target_key)

    def process_data(self, data):
        """Shared data handling for both sub-engines."""
        super().process_data(data)
        self.classification_engine.data = self.data
        self.regression_engine.data = self.data

    def process_target(self, data):
        self.classification_engine.process_target(data)
        self.regression_engine.process_target(data)

    def compute_outputs(self):
        """
        Split the model output into regression and classification parts and calculate metrics of each.
        Assumes model output shape: [batch, regression_size + num_classes] where the first part corresponds to all
        regression predictions combined and the second to classification logits.
        """
        if self.regression_size is None:
            self.regression_size = sum(self.regression_engine.target_sizes)
        self.regression_engine.model_out = self.model_out[:, :self.regression_size]
        self.classification_engine.model_out = self.model_out[:, self.regression_size:]
        regression_outputs = self.regression_engine.compute_outputs()
        classification_outputs = self.classification_engine.compute_outputs()
        return regression_outputs | classification_outputs

    def compute_metrics(self):
        """Compute regression and classification losses with weighting"""
        # Compute each loss separately
        reg_metrics = self.regression_engine.compute_metrics()
        cls_metrics = self.classification_engine.compute_metrics()

        reg_loss = reg_metrics.pop("loss")
        cls_loss = cls_metrics.pop("loss")

        metrics = {
            **reg_metrics,
            **cls_metrics,
            "regression_loss": reg_loss,
            "classification_loss": cls_loss,
        }

        if self.loss_weight is None:
            # Uncertainty-based combination
            reg_log_var = self.module.log_vars["regression_log_var"]
            cls_log_var = self.module.log_vars["classification_log_var"]
            weighted_reg_loss = 0.5*torch.exp(-reg_log_var) * reg_loss + 0.5*reg_log_var
            weighted_cls_loss = torch.exp(-cls_log_var) * cls_loss + 0.5*cls_log_var
            self.loss = weighted_reg_loss + weighted_cls_loss
            for k, v in self.module.log_vars.items():
                metrics[k] = v.detach()
        else:
            self.loss = self.loss_weight * reg_loss + (1 - self.loss_weight) * cls_loss

        metrics["loss"] = self.loss
        return metrics

    def save_state(self, suffix="", name=None):
        self.state_data["target_sizes"] = self.regression_engine.target_sizes
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
        if "target_sizes" in self.state_data:
            self.regression_engine.target_sizes = self.state_data["target_sizes"]
