
# basic imports
import numpy as np

# Nn imports
import torch

# watchmal imports
from watchmal.engine.graph.reconstruction import ReconstructionEngine

from watchmal.utils.logging_utils_caverns import setup_logging

log = setup_logging(__name__)


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(
        self, 
        target_key, 
        model, 
        rank, 
        device, 
        dump_path, 
        wandb_run=None,
        dataset=None,
        output_center=0, 
        output_scale=1
    ):
        """
        Parameters
        ==========
        target_key : string
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
        super().__init__(
            target_key, 
            model, 
            rank, 
            device, 
            dump_path,
            wandb_run=wandb_run,
            dataset=dataset
        )
        

        self.output_center = output_center # define for the cnn. No idea when it is used
        self.output_scale = output_scale   # neither why to do scaling this way


    def make_plots(self, preds, targets, prefix_plot_name):
        # Plotting utilities live in a separate WatChMaL viz repo and are not shipped
        # here; import lazily so a run without them degrades to "no plots" rather than
        # failing to import the engine.
        try:
            from watchmal.utils.viz_utils import preds_targets_histogram
        except ImportError:
            log.warning("viz_utils not available; skipping evaluation plots.")
            return

        # Denormalizing data
        # index 0 is maximum values, index 1 is minimum
        
        if self.target_norm is not None:
            preds =  self.target_norm[1] + (self.target_norm[0] - self.target_norm[1]) * preds
            targets = self.target_norm[1] + (self.target_norm[0] - self.target_norm[1]) * targets
        
        # Plots
        for i in range(len(self.target_names)):

            i_preds       = preds[:, i]
            i_targets     = targets[:, i]
            i_target_name = self.target_names[i]

            plot_name = prefix_plot_name + f"tgt_vs_out_{i_target_name}"
            preds_targets_histogram(
                self.wandb_run,
                i_preds,
                i_targets,
                target_name=i_target_name,
                fill=False,
                element='step',
                log_yscale=False,
                folder_path=self.dump_path,
                plot_name=plot_name
            )

    def to_disk_data_reformat(self, preds, targets, indices):

        # Expecting (batch_size, len(target_names)) as output shape
        # target_names enable multi-dim target supports (both classification or regression)
        
        preds = np.concatenate(preds).reshape(-1, len(self.target_names))
        targets = np.concatenate(targets).reshape(-1, len(self.target_names))
 
        res = {'preds': preds,'targets': targets}


        # 1. Concatenate all batch arrays into one long array.
        combined_preds = np.concatenate(preds, axis=0)
        combined_targets = np.concatenate(targets, axis=0)
        
        # 2. Reshape to the final desired shape: (total_samples, num_targets)
        # (This handles the scalar case by turning a (N,) array into (N, 1).)
        final_preds = combined_preds.reshape(-1, len(self.target_names))
        final_targets = combined_targets.reshape(-1, len(self.target_names))

        res = {'preds': final_preds, 'targets': final_targets}

        # Indices are 1D, so concatenating them is equivalent to flattening.
        if indices is not None:
            final_indices = np.array(indices).flatten()
            res['indices'] = final_indices
            
        return res



    def forward(self, forward_type='train'):
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

        metrics = {}
        outputs = {}
        grad_enabled = True if forward_type == 'train' else False

        with torch.set_grad_enabled(grad_enabled):
            # handles the extra parameter generated by the model for multitask loss
            out = self.model(self.data)

            # If the model returns multiple elements, unpack
            if isinstance(out, (tuple, list)) and len(out) > 1:
                model_out, log_vars = out
                model_out = torch.cat(model_out, dim=1)
            else:
                model_out = out
                log_vars = None  # or a default tensor if needed

            # Call loss, pass log_vars if the loss expects it
            head_losses = {}
            if log_vars is not None:
                loss, head_losses = self.criterion(model_out, self.target, log_vars)
                ## fix for issues with DDP, forces parameter updates even if unused
                loss = loss + 0.0 * log_vars.sum()
            else:
                loss = self.criterion(model_out, self.target)

            # Log data about the outputs
            outputs['pred_min']    = model_out.min()
            outputs['pred_max']    = model_out.max()
            outputs['pred_mean']   = model_out.mean()
            outputs['pred_median'] = model_out.median()

            # Log data about the metrics
            metrics['loss']     = loss
            metrics.update({f'loss_{name}': v for name, v in head_losses.items()})

            if forward_type == 'test':
                outputs['pred'] = model_out

        # metrics and potentially outputs are still on the computational graph        
        return outputs, metrics

    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
