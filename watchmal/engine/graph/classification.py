import torch
import numpy as np
import pandas as pd

import scipy.special as special

# watchmal imports
from watchmal.engine.graph.reconstruction import ReconstructionEngine

from watchmal.utils.logging_utils_caverns import setup_logging
from watchmal.utils.viz_utils import roc_curve, p_r_curve, confusion_matrix, scatplot_2d, combined_histograms_plot, histogram_2d, count_plot, zoomed_roc_curve

log = setup_logging(__name__)

class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(
            self, 
            target_key, 
            model, 
            rank, 
            device, 
            dump_path,
            wandb_run=None, 
            dataset=None,
            flatten_model_output=False, 
            prediction_threshold=None,
        ):
        """
        Parameters
        ==========
        target_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
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
        
        self.flatten_model_output = flatten_model_output
        self.prediction_threshold = prediction_threshold

        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

        self.signal_key = ""
        self.label_set = []


    def set_dataset(self, dataset, dataset_config): # backward compatibility only
        super().set_dataset(dataset, dataset_config)
        self.signal_key = dataset_config.signal_key
        
        # Get the label_set
        for trf in dataset.transforms.transforms: # dataset.transform is a T.Compose() object, to access the list of transform calling .transforms is needed
            if trf.__class__.__name__ == 'MapLabels':
                self.label_set = trf.label_set
                break
        
    def configure_dataset(self, data_config):
        """
        Configure PyG dataset from data_config.
        Slight overwrite to handle "signal_key" (for the plots)
        in case of classification tasks.
        """
        super().configure_dataset(data_config)

        self.signal_key = data_config.dataset.signal_key


    def make_plots(self, preds, targets, prefix_plot_name):
        """
        Only rank 0 should call make_plots()
        """
      
        softmax_preds = special.softmax(preds, axis=1)
        predicted_classes = np.argmax(softmax_preds, axis=1)

        raw_columns = ['raw_preds_' + name  for name in self.target_names]
        sft_columns = ['softmax_preds_' + name for name in self.target_names]

        data = pd.DataFrame({})
        data['target_names'] = [self.target_names[int(i)] for i in targets]

        for raw_name, softmax_name, i in zip(raw_columns, sft_columns, range(len(self.target_names))):
            data[raw_name]     = preds[:, i]
            data[softmax_name] = softmax_preds[:, i]


        # Caution : raw_preds is [a, b] right now, and we consider raw_preds[1] as the signal
        roc_curve(
            self.wandb_run,
            softmax_preds=softmax_preds,
            targets=targets,
            target_names=self.target_names,
            signal_key=self.signal_key,
            folder_path=self.dump_path,
            plot_name=prefix_plot_name + 'roc_curve',
            log_scale=False,
            figsize=(8, 8)
        )

        confusion_matrix(
            self.wandb_run,
            predicted_classes=predicted_classes,
            targets=targets,
            target_names=self.target_names,
            signal_key=self.signal_key,
            folder_path=self.dump_path,
            plot_name=prefix_plot_name + 'cf_matrix',
        )

    def metric_data_reformat(self, loss, accuracy):
        """
        If new metrics are added in the classification forward loop, they need to be added here too.
        We keep the argument as a a, b, b and not **kwargs to prevent no support of new metrics
        """

        loss = np.array(loss).flatten()
        accuracy = np.array(accuracy).flatten()

        res = {'loss': loss, 'accuracy': accuracy}
        return res

    def to_disk_data_reformat(self, preds, targets, indices=None):

        preds   = np.array(preds).reshape(-1, len(self.target_names))
        targets = np.array(targets).flatten()
        res = {'preds': preds,'targets': targets}

        if indices is not None:
            final_indices = np.array(indices).flatten()
            res['indices'] = final_indices
            
        return res


    def forward(self, forward_type='train'):
        """
        Compute predictions and metrics for a batch of data.

        Parameters
        ==========
        forward_type : (str) either 'train', 'val' or 'test'
            Whether in training mode, requiring computing gradients for backpropagation
            For 'test' also returns the softmax value in outputs
        Returns
        =======
        dict
            Dictionary containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        metrics = {}
        outputs = {}
        grad_enabled = True if forward_type == 'train' else False

        with torch.set_grad_enabled(grad_enabled):
    
            model_out = self.model(self.data) # even in ddp, the forward is done with self.model and not self.module
            
            # Compute the loss
            if self.flatten_model_output:
                model_out = torch.flatten(model_out)

            self.target = self.target.reshape(-1)
            loss = self.criterion(model_out, self.target)

            # Apply softmax to model_out
            if self.flatten_model_output:
                softmax = self.sigmoid(model_out)
            else: 
                softmax = self.softmax(model_out)

            # Compute accuracy based on the softmax values
            if self.flatten_model_output:
                preds = ( softmax >= self.prediction_threshold )
            else :
                preds = torch.argmax(model_out, dim=-1)
            
            accuracy = (preds == self.target).sum() / len(self.target)

            # Add the metrics to the output dictionary
            metrics['loss']     = loss
            metrics['accuracy'] = accuracy

            # Note : this softmax saving will be modified. Even maybe deleted
            if forward_type == 'test': # In testing mode we also save the softmax values
                outputs['pred'] = model_out

        # metrics and potentially outputs contains tensors linked to the gradient graph (and on gpu if any) 
        return outputs, metrics


        # if needed one day : predicted_labels.nelement() see https://pytorch.org/docs/stable/generated/torch.numel.html#torch.numel (nelement is an alias for .numel())