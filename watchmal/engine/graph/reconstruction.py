"""
Graph reconstruction engine: training/evaluation with dict batches and optional wandb/early-stopping.
Inherits from BaseEngine.
"""

# generic imports
import numpy as np
from datetime import datetime
from abc import abstractmethod

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch

# WatChMaL imports
from watchmal.dataset.samplers.samplers import SubsetSequentialSampler
from watchmal.utils.logging_utils_caverns import setup_logging
from watchmal.utils.early_stopping import EarlyStopping

from watchmal.engine.base_engine import BaseEngine

log = setup_logging(__name__)


class ReconstructionEngine(BaseEngine):
    def __init__(
        self,
        target_key,
        model,
        rank,
        device,
        dump_path,
        wandb_run=None,
        dataset=None,
    ):
        """
        Parameters
        ==========
        target_key : str
            Name of the key for the target values in the dictionary returned by the dataloader.
        model : nn.Module
            Full network used for training or evaluation.
        rank : int
            Rank of this process (in multiprocessing mode).
        device : int or torch.device
            Device to run on.
        dump_path : str
            Path to store outputs (should end with '/').
        wandb_run : optional
            If set, used for logging and save_state artifacts.
        dataset : optional
            Pre-built dataset (for graph-style pipeline).
        """
        super().__init__(
            target_key=target_key,
            model=model,
            rank=rank,
            device=device,
            dump_path=dump_path,
            wandb_run=wandb_run,
            dataset=dataset,
        )
        self.early_stopping = None

    @abstractmethod
    def forward(self, forward_type):
        pass 

    @abstractmethod
    def to_disk_data_reformat(self):
        pass

    @abstractmethod
    def make_plots(self):
        pass


    def configure_early_stopping(self, early_stopping_config):
        self.early_stopping = instantiate(early_stopping_config)

    def configure_dataset(self, data_config): 
        """
        Configure PyG dataset from data_config.
        Handles both pre-built (in_memory) and per-engine instantiation cases.
        Sets self.dataset, self.split_path, self.is_pyg, and self.target_names.
        
        Parameters
        ----------
        data_config : DictConfig or dict
            Configuration containing:
            - data.dataset.dataset_parameters: PyG dataset parameters
            - data.dataset.split_path: path to split indices file
            - data.dataset.target_names: list of target names
            - data.transforms: transforms config
            - kind: string indicating dataset kind (to determine if PyG)
        """
        # Extract metadata from config
        dataset_config = data_config.dataset
        self.split_path = dataset_config.split_path
        self.target_names = list(dataset_config.target_names)
        self.is_pyg = True  # Graph datasets are always PyG
        
        # Check if dataset was pre-built (in_memory case)
        if self.dataset is not None:
            # Dataset already built in main.py (in_memory case)
            log.info(f"Using pre-built dataset (in_memory). Length: {len(self.dataset)}")
        else:
            # Build dataset per engine (non-in_memory case)
            from watchmal.dataset.graph.data_utils import get_dataset
            log.info(f"Building PyG dataset per engine")
            self.dataset = get_dataset(data_config)
        
        # Graph-specific: feat_norm / target_norm from Normalize transform (for display)
        ft_norm, target_norm = None, None
        if hasattr(self.dataset, 'transforms') and hasattr(self.dataset.transforms, 'transforms'):
            for trf in self.dataset.transforms.transforms:
                if trf.__class__.__name__ == "Normalize":
                    ft_norm, target_norm = trf.feat_norm, trf.target_norm
                    break
        self.feat_norm, self.target_norm = ft_norm, target_norm
        
    def set_dataset(self, dataset, dataset_config):
        """
        Set dataset directly (backward compatibility).
        Note: configure_dataset() is now the preferred way to configure datasets.
        """
        super().set_dataset(dataset, dataset_config)
        # Graph-specific: feat_norm / target_norm from Normalize transform (for display)
        
        ft_norm, target_norm = None, None
        for trf in dataset.transforms.transforms:
            if trf.__class__.__name__ == "Normalize":
                ft_norm, target_norm = trf.feat_norm, trf.target_norm
                break
        self.feat_norm, self.target_norm = ft_norm, target_norm

    @staticmethod
    def _accumulate_metrics(metrics_epoch_history, metrics):
        """Add batch metrics into epoch accumulator (in-place). metrics values are scalars."""
        for k in metrics:
            if k not in metrics_epoch_history:
                metrics_epoch_history[k] = 0.0
            metrics_epoch_history[k] += metrics[k]

    def _mount_batch(self, batch):
        """
        Set self.data and self.target from batch (hierarchical pmt_data/mpmt_data or flat data/target_key).
        Returns batch indices tensor if present (e.g. for evaluate), else None.
        """
        if "mpmt_data" in batch:
            pmt_batch = batch["pmt_data"]
            mpmt_batch = batch["mpmt_data"]
            self.data = {
                "pmt_data": {
                    "data": pmt_batch["data"].to(self.device),
                    "target": pmt_batch["target"].to(self.device),
                },
                "mpmt_data": mpmt_batch.to(self.device),
            }
            self.target = pmt_batch["target"].to(self.device)
            indice = pmt_batch.get("indice")
        else:
            self.data = batch["data"].to(self.device)
            self.target = batch[self.target_key].to(self.device)
            indice = batch.get("indice")
        return indice.to(self.device) if indice is not None else None

    def sub_train(self, loader, val_interval):
        """
        Each process performs its own (train and) sub_train. Outputs are gathered in the main train() loop.
        """
        self.model.train()
        assert len(loader) >= 1, "Train loader must have at least one batch"

        metrics_epoch_history = {"loss": 0}

        for step, train_data in enumerate(loader):
            self._mount_batch(train_data)

            # Call forward: make a prediction & measure the average error using data = self.data
            outputs, metrics  = self.forward(forward_type='train')
            
            # Call backward: back-propagate error and update weights using loss = self.loss
            self.loss = metrics['loss']
            self.backward()

            # # Run scheduler # for now we decide to apply step after every epoch step (and not batch step)
            # if self.scheduler is not None:
            #     self.scheduler.step()
            
            # If not detaching now ( with .item() ) all the data of the epoch will be load into GPU memory
            # v.item() converts torch.tensors to python floats (and detachs + moves to cpu)
            metrics = {k: v.item() for k, v in metrics.items()} 
            outputs = {k: v.item() for k, v in outputs.items()}

            # For now we only monitor rank 0
            if self.rank == 0:

                # --- Logs in wandb --- #
                if self.wandb_run is not None:

                    self.wandb_run.log(
                        {'train_batch_' + k: v for k, v in outputs.items()} | 
                        {'train_batch_' + k: v for k, v in metrics.items()}
                    )

                # --- Keep track of metrics --- #
                self._accumulate_metrics(metrics_epoch_history, metrics)

            # --- Display --- #
            if ( step  % val_interval == 0 ):
                if ( self.rank == 0 ) :
                    log.info(f"GPU : {self.device} | Steps : {step + 1}/{len(loader)} | Iteration : {self.iteration + step} | Batch Size : {loader.batch_size}")
                    log.info(f"Batch metrics {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())}")
                    
        
        self.iteration += ( step + 1 )

        if self.rank == 0:
            # Mean over the epoch for each metric
            for k in metrics_epoch_history.keys():
                metrics_epoch_history[k] /= (step + 1)

        return metrics_epoch_history

    def sub_validate(self, loader, forward_type='val'):
        """
        loader: Loader on which to perform the validation.
        forward_type: forward_type argument to pass to the forward method.
            If set to 'test' the output will be two dictionnaries :
            One containing the metrics, one containing the raw preds + targets
            Otherwise it just return the metrics dictionary and raw_preds and
            targets are not stored.
        """

        metrics_epoch_history = {'loss': 0}  # loss is mandatory for Class and Reg engine
        to_disk_epoch_history = {'preds': [], 'targets': []}  # Will be saved in numpy arrays
        wandb_prefix = "test_" if forward_type == "test" else ""
            
        self.model.eval()
        assert len(loader) >= 1, "Validation loader must have at least one batch"
        with torch.no_grad():
            for step, val_batch in enumerate(loader):
                self._mount_batch(val_batch)

                # Evaluate the network for this batch
                outputs, metrics = self.forward(forward_type=forward_type) # output is a dictionary with torch.tensors
                
                # In case of ddp we reduce outputs to get the global performance
                # Note : It is currently done at each step to optimize gpu memory usage
                # But this could also be perform at the end of the validation epoch
                
                if forward_type == 'test':
                    preds = {'pred': outputs.pop('pred')}

                if self.is_distributed:
                    metrics = self.get_reduced(metrics)
                    outputs = self.get_reduced(outputs)

                    if forward_type == 'test':
                        self.target = self.get_gathered(self.target)
                        preds = {'pred': self.get_gathered(preds['pred'])}
                
                # Output of get_gathered will be None for rank != 0
                # So we only need to detach rank 0 tensors
                if self.rank == 0: 
                
                    # Detaching outputs tensors ( with .item() )
                    # otherwise all the data of the epoch will be load into GPU memory
                    # v.item() converts torch.tensors to python floats (and detachs + moves to cpu)
                    metrics = {k: v.item() for k, v in metrics.items()} 
                    outputs = {k: v.item() for k, v in outputs.items()} 

                    if forward_type == 'test':
                        self.target = self.target.detach().cpu().numpy() # we store them for roc curve etc..
                        preds   = {k: v.detach().cpu().numpy() for k, v in preds.items()}

                    # --- Storing for saving --- # 
                    if forward_type == 'test':
                        to_disk_epoch_history['targets'].append(self.target)
                        to_disk_epoch_history['preds'].append(preds['pred']) 

                    # --- Storing global performances --- #
                    self._accumulate_metrics(metrics_epoch_history, metrics)

                    # --- Logs --- #
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {wandb_prefix + 'val_batch_' + k: v for k, v in outputs.items()} | 
                            {wandb_prefix + 'val_batch_' + k: v for k, v in metrics.items()}
                        )

            # Take the mean over the epoch 
            if self.rank == 0:
                for k in metrics_epoch_history.keys():
                    metrics_epoch_history[k] /= (step + 1)
        
        if forward_type == 'test':
            return metrics_epoch_history, to_disk_epoch_history
        
        return metrics_epoch_history

    def train(self, epochs=0, val_interval=20, checkpointing=False):
        """
        Train the model on the training set. The best state is always saved during training.

        Parameters
        ==========
        epochs: int
            Number of epochs to train, default 1
        val_interval: int
            Number of iterations between each validation, default 20
        checkpointing: bool
            Whether to save state every validation, default False
        """
        
        
        start_run_time = datetime.now()
        #log.info(f"Engine : {self.rank} | Dataloaders : {self.data_loaders}")
        if self.rank == 0:
            log.info( f"\033[1;96m********** 🚀 Starting training for {epochs} epochs 🚀 **********\033[0m")
        

        # initialize epoch and iteration counters
        #epoch               = 0 # (used by nick)  counter of epoch
        self.iteration       = 1 # (used by erwan) counter of the steps of all epochs
        
        self.best_training_loss   = np.inf
        self.best_validation_loss = np.inf
        

        train_loader = self.data_loaders["train"]
        val_loader   = self.data_loaders["validation"]

        # Watching 
        if self.wandb_run is not None:
            self.wandb_run.watch(self.module, log='all', log_freq=val_interval * 2, log_graph=True)
            self.wandb_run.log({'max_datapoints_seen': epochs * len(train_loader) * train_loader.batch_size})

        # global loop for multiple epochs        
        for epoch in range(epochs):
            
            # variables that will be used outside the train function
            self.epoch = epoch

            # ---- Starting the training epoch ---- #
            epoch_start_time = datetime.now()
            if ( self.rank == 0 ):
                log.info(f"\n\nTraining epoch {self.epoch + 1}/{epochs} starting at {epoch_start_time}")
            

            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            log.info(f"train_loader: {train_loader}")
            metrics_epoch_history = self.sub_train(train_loader, val_interval) # one train epoch.
            
            # Run scheduler
            if ( self.scheduler is not None ) and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                    
                if ( self.scheduler.get_last_lr() != current_lr ):
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")

            epoch_end_time = datetime.now()

            # --- Display global info about the train epoch --- #
            if self.rank == 0:
                log.info(f"(Train) Epoch : {epoch + 1} completed in {(epoch_end_time - epoch_start_time)} | Iteration : {self.iteration} ")
                log.info(f"Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info(f"Metrics over the (train) epoch {', '.join(f'{k}: {v:.5g}' for k, v in metrics_epoch_history.items())}")

                # --- Wandb logs --- #
                if self.wandb_run is not None:

                    self.wandb_run.log({'epoch': epoch}) # To monitor the number of epoch step (in the training fail in the middle of the run)                                        
                    self.wandb_run.log(
                        {'train_epoch_' + k: v for k, v in metrics_epoch_history.items()}
                    )

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.wandb_run.log({'learning_rate': self.optimizer.param_groups[0]['lr']}) # Changing the optimizer might lead to a change of this line too.    
                        else :
                            self.wandb_run.log({'learning_rate': self.scheduler.get_last_lr()[0]}) # Changing the optimizer might lead to a change of this line too.

                    if metrics_epoch_history['loss'] < self.best_training_loss:
                        self.best_training_loss = metrics_epoch_history['loss']
                        self.wandb_run.log({'best_train_epoch_loss': self.best_training_loss})

            
            # ---- Starting the validation epoch ---- #
            epoch_start_time = datetime.now()
            if ( self.rank == 0 ):
                log.info("")
                log.info(f" -- Validation epoch starting at {epoch_start_time}")

            metrics_epoch_history = self.sub_validate(val_loader, forward_type="val")

            # --- Early Stopping --- #
            if self.early_stopping is not None:
                stop_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
                if self.rank == 0:
                    self.early_stopping(metrics_epoch_history['loss'])
                    stop_flag.fill_(int(self.early_stopping.should_stop))

                if self.is_distributed:
                        torch.distributed.broadcast(stop_flag, src=0) # rank 0 (src) pushes flag across the network of processes
                
            # --- Scheduler --- #
            if ( self.scheduler is not None ) and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):

                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(metrics_epoch_history['loss'])

                if ( self.optimizer.param_groups[0]['lr'] != current_lr ):
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")


            epoch_end_time = datetime.now()

            # --- Display global info about the validation epoch --- #
            if self.rank == 0:
                log.info(f" -- Validation epoch completed in {epoch_end_time - epoch_start_time} | Iteration : {self.iteration}")
                log.info(f" -- Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info(f" -- Metrics over the (val) epoch {', '.join(f'{k}: {v:.5g}' for k, v in metrics_epoch_history.items())}")


                # --- Wandb --- #
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {'val_epoch_' + k: v for k, v in metrics_epoch_history.items()}
                    )

                # --- Model Saving --- #
                if checkpointing: # if checkpointing the model is saved at the end of each validation epoch
                    self.save_state()
                            
                if metrics_epoch_history["loss"] < self.best_validation_loss:
                    log.info(" ... Best validation loss so far!")
                    self.best_validation_loss = metrics_epoch_history["loss"]
                    self.save_state(suffix="_BEST")


            # --- Early stopping --- #
            if self.early_stopping is not None:
                if stop_flag.item():
                    if self.rank == 0:
                        log.info("Early stopping triggered.")
                        self.wandb_run.log({'early_stopped': True})
                    if self.is_distributed: # Ensure we stop on all processes. Barrier has to be called from all of the network (processes)
                        torch.distributed.barrier()
                    break
 

    def validate(self, data_loader_name, forward_type, prefix_plot_name):

        val_loader = self.data_loaders[data_loader_name]
        start_time = datetime.now()

        if self.rank == 0:
            log.info(f"\n\n Validation starting")


        metrics_epoch_history = self.sub_validate(val_loader, forward_type=forward_type)
        
        if forward_type == 'test':
            metrics_epoch_history, to_disk_epoch_history = metrics_epoch_history

            if self.rank == 0:

                self.make_plots(
                    prefix_plot_name,
                    targets=to_disk_epoch_history["targets"],
                    preds=to_disk_epoch_history["preds"],
                )


        end_time = datetime.now()
        if self.rank == 0:
            log.info(f"Validation completed in {end_time - start_time} | Iteration check : {self.iteration}")
            log.info(f"Total time since the beginning of the validation : {end_time - start_time}")
            log.info(f"Metrics over the (val) epoch {', '.join(f'{k}: {v:.5g}' for k, v in metrics_epoch_history.items())}")



    def evaluate(self, prefix_for_plot_names, report_interval=20, batch_log=False):
        """Evaluate the performance of the trained model on the test set."""
                
        if self.rank == 0:
            log.info(f"\n\nTest epoch starting.\nOutput directory: {self.dump_path}")

        loader = self.data_loaders["test"]
        assert len(loader) >= 1, "Test loader must have at least one batch"

        with torch.no_grad():
            self.model.eval()
            metrics_epoch_history = {"loss": 0.0}
            preds_list = []
            targets_list = []
            indices_list = []

            # sampler = loader.sampler.sampler if self.is_distributed else loader.sampler
            sampler = loader.sampler
            if not isinstance(sampler, SubsetSequentialSampler):
                raise ValueError(f"For test run sampler should only be of type 'SequentialSampler', got {type(sampler)}")
            
            start_time = datetime.now()

            if self.rank == 0:
                log.info(f"Engine : {self.rank} | Sampler len  : {len(loader.sampler)}")
                log.info(f"Engine : {self.rank} | Test Dataloader bs : {loader.batch_size}")   
                log.info(f"Engine : {self.rank} | Nb steps           : {len(loader)}")

            for step, eval_data in enumerate(loader):
                batch_indices = self._mount_batch(eval_data)

                outputs, metrics = self.forward(forward_type="test")
                preds = {'pred': outputs.pop('pred')}

                metrics = {k: v.item() for k, v in metrics.items()}
                outputs = {k: v.item() for k, v in outputs.items()}

                preds_np = {k: v.detach().cpu().numpy() for k, v in preds.items()}
                targets_np = self.target.detach().cpu().numpy()
                indices_cpu = batch_indices.detach().cpu()

                if self.wandb_run is not None and batch_log:
                    self.wandb_run.log(
                        {'test_batch_' + k: v for k, v in metrics.items()} |
                        {'test_batch_' + k: v for k, v in outputs.items()}
                    )

                self._accumulate_metrics(metrics_epoch_history, metrics)

                preds_list.append(preds_np["pred"])
                targets_list.append(targets_np)
                indices_list.append(indices_cpu)

                if step % report_interval == 0 and self.rank >= 0:
                    log.info(
                        f"  Step {step + 1}/{len(loader)}"
                        f"  Evaluation {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())},"
                    )

        end_time = datetime.now()

        # Combine lists into full tensors
        local_preds = np.concatenate(preds_list, axis=0)
        local_targets = np.concatenate(targets_list, axis=0)
        local_indices = torch.cat(indices_list, dim=0).to(self.device)

        # Gather across all GPUs
        global_preds = self.get_gathered(torch.tensor(local_preds).to(self.device))
        global_targets = self.get_gathered(torch.tensor(local_targets).to(self.device))
        global_indices = self.get_gathered(local_indices)

        if self.rank == 0:
            for k in metrics_epoch_history:
                metrics_epoch_history[k] /= step + 1

            log.info(f"Evaluation total time {end_time - start_time}")
            log.info(f"Average time per event: {(end_time - start_time) / len(loader.sampler) }")
            log.info(f"Metrics over the test epoch {', '.join(f'{k}: {v:.5g}' for k, v in metrics_epoch_history.items())}")

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {'test_epoch_' + k: v for k, v in metrics_epoch_history.items()}
                )

            # Convert gathered tensors to numpy arrays
            global_preds = global_preds.cpu().numpy()
            global_targets = global_targets.cpu().numpy()
            global_indices = global_indices.cpu().numpy()

            # deduplicate based on indices (in case of overlapping subsets)
            _, unique_idx = np.unique(global_indices, return_index=True)
            global_indices = global_indices[unique_idx]
            global_preds = global_preds[unique_idx]
            global_targets = global_targets[unique_idx]

            # Reformat and save
            to_disk_epoch_history = {
                'preds': global_preds,
                'targets': global_targets,
                'indices': global_indices,
            }
            to_disk_epoch_history = self.to_disk_data_reformat(**to_disk_epoch_history)

            log.info("Saving the data...")
            for k, v in to_disk_epoch_history.items():
                save_path = self.dump_path + k + ".npy"
                np.save(save_path, v)
                if self.wandb_run is not None:
                    self.wandb_run.save(save_path)
            log.info("Done")

            log.info(f"Starting to compute plots..")
            self.make_plots(
                prefix_plot_name=prefix_for_plot_names,
                targets=to_disk_epoch_history['targets'],
                preds=to_disk_epoch_history['preds'],
            )     
            log.info("Done")