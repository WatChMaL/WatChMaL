"""
Class for training a fully supervised classifier
"""

# generic imports
import numpy as np
from time import strftime, localtime, time
from datetime import timedelta
from abc import ABC, abstractmethod
import logging

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVLog

log = logging.getLogger(__name__)


class ReconstructionEngine(ABC):
    def __init__(self, truth_key, model, rank, gpu, dump_path):
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
        """
        # create the directory for saving the log and dump files
        self.epoch = 0
        self.step = 0
        self.iteration = 0
        self.best_validation_loss = 1.0e10
        self.dump_path = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)
        self.truth_key = truth_key

        # Set up the parameters to save given the model type
        if isinstance(self.model, DistributedDataParallel):
            self.is_distributed = True
            self.module = self.model.module
            self.n_gpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.module = self.model

        self.data_loaders = {}

        # define the placeholder attributes
        self.data = None
        self.target = None
        self.loss = None

        # logging attributes
        self.train_log = CSVLog(self.dump_path + f"log_train_{self.rank}.csv")

        if self.rank == 0:
            self.val_log = CSVLog(self.dump_path + "log_val.csv")

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config)

    def configure_optimizers(self, optimizer_config):
        """Instantiate an optimizer from a hydra config."""
        self.optimizer = instantiate(optimizer_config, params=self.module.parameters())

    def configure_scheduler(self, scheduler_config):
        """Instantiate a scheduler from a hydra config."""
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

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
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)

    def get_synchronized_outputs(self, output_dict):
        """
        Gathers results from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        output_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_output_dict : dict of torch.Tensor
            Dictionary containing concatenated tensor values gathered from all processes
        """
        global_output_dict = {}
        for name, tensor in output_dict.items():
            if self.is_distributed:
                if self.rank == 0:
                    tensor_list = [torch.zeros_like(tensor, device=self.device) for _ in range(self.n_gpus)]
                    torch.distributed.gather(tensor, tensor_list)
                    global_output_dict[name] = torch.cat(tensor_list).detach().cpu().numpy()
                else:
                    torch.distributed.gather(tensor, dst=0)
            else:
                global_output_dict[name] = tensor.detach().cpu().numpy()
        return global_output_dict

    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_metric_dict : dict
            Dictionary containing mean of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, tensor in zip(metric_dict.keys(), metric_dict.values()):
            if self.is_distributed:
                torch.distributed.reduce(tensor, 0)
                if self.rank == 0:
                    global_metric_dict[name] = tensor.item()/self.n_gpus
            else:
                global_metric_dict[name] = tensor.item()
        return global_metric_dict

    @abstractmethod
    def forward(self, train=True):
        pass

    def backward(self):
        """Backward pass using the loss computed for a mini-batch"""
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()  # compute new gradient
        self.optimizer.step()  # step params

    def train(self, epochs=0, val_interval=20, num_val_batches=4, checkpointing=False, save_interval=None):
        """
        Train the model on the training set. The best state is always saved during training.

        Parameters
        ==========
        epochs: int
            Number of epochs to train, default 1
        val_interval: int
            Number of iterations between each validation, default 20
        num_val_batches: int
            Number of mini-batches in each validation, default 4
        checkpointing: bool
            Whether to save state every validation, default False
        save_interval: int
            Number of epochs between each state save, by default don't save
        """
        if self.rank == 0:
            log.info(f"Training {epochs} epochs with {num_val_batches}-batch validation each {val_interval} iterations")
        # set model to training mode
        self.model.train()
        # initialize epoch and iteration counters
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        # keep track of the validation loss
        self.best_validation_loss = np.inf
        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])
        # global training loop for multiple epochs
        start_time = time()
        step_time = start_time
        epoch_start_time = start_time
        for self.epoch in range(epochs):
            if self.rank == 0:
                if self.epoch > 0:
                    log.info(f"Epoch {self.epoch} completed in {timedelta(seconds=time() - epoch_start_time)}")
                    epoch_start_time = time()
                log.info('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            train_loader = self.data_loaders["train"]
            self.step = 0
            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)
            # local training loop for batches in a single epoch
            steps_per_epoch = len(train_loader)
            for self.step, train_data in enumerate(train_loader):
                # Train on batch
                self.data = train_data['data'].to(self.device)
                self.target = train_data[self.truth_key].to(self.device)
                # Call forward: make a prediction & measure the average error using data = self.data
                outputs, metrics = self.forward(True)
                # Call backward: back-propagate error and update weights using loss = self.loss
                self.backward()
                # run scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                # update the epoch and iteration
                self.step += 1
                self.iteration += 1
                # get relevant attributes of result for logging
                log_entries = {"iteration": self.iteration, "epoch": self.epoch, **metrics}
                # record the metrics for the mini-batch in the log
                self.train_log.log(log_entries)
                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    if self.rank == 0:
                        previous_step_time = step_time
                        step_time = time()
                        average_step_time = (step_time - previous_step_time)/val_interval
                        print(f"Iteration {self.iteration}, Epoch {self.epoch+1}/{epochs}, Step {self.step}/{steps_per_epoch}"
                              f" Step time {timedelta(seconds=average_step_time)},"
                              f" Epoch time {timedelta(seconds=step_time-epoch_start_time)}"
                              f" Total time {timedelta(seconds=step_time-start_time)}")
                        print(f"  Training   {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())},", end="")
                    self.validate(val_iter, num_val_batches, checkpointing)
            # save state at end of epoch
            if self.rank == 0 and (save_interval is not None) and ((self.epoch+1) % save_interval == 0):
                self.save_state(suffix=f'_epoch_{self.epoch+1}')
        self.train_log.close()
        if self.rank == 0:
            log.info(f"Epoch {self.epoch} completed in {timedelta(seconds=time() - epoch_start_time)}")
            log.info(f"Training {epochs} epochs completed in {timedelta(seconds=time()-start_time)}")
            self.val_log.close()

    def validate(self, val_iter, num_val_batches, checkpointing):
        """
        Perform validation with the current state, on a number of batches of the validation set.

        Parameters
        ----------
        val_iter : iter
            Iterator of the validation dataset.
        num_val_batches : int
            Number of validation batches to iterate over.
        checkpointing : bool
            Whether to save the current state to disk.
        """
        # set model to eval mode
        self.model.eval()
        val_metrics = None
        for val_batch in range(num_val_batches):
            # get validation data mini-batch
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                if self.is_distributed:
                    self.data_loaders["validation"].sampler.set_epoch(self.data_loaders["validation"].sampler.epoch+1)
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)
            # extract the event data and target from the input data dict
            self.data = val_data['data'].to(self.device)
            self.target = val_data[self.truth_key].to(self.device)
            # evaluate the network
            outputs, metrics = self.forward(False)
            if val_metrics is None:
                val_metrics = metrics
            else:
                for k, v in metrics.items():
                    val_metrics[k] += v
        # record the validation stats to the csv
        val_metrics = {k: v/num_val_batches for k, v in val_metrics.items()}
        val_metrics = self.get_synchronized_metrics(val_metrics)
        if self.rank == 0:
            log_entries = {"Iteration": self.iteration, "epoch": self.epoch, **val_metrics, "saved_best": False}
            print(f"  Validation {', '.join(f'{k}: {v:.5g}' for k, v in val_metrics.items())}")
            # Save if this is the best model so far
            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                log.info(f"Best validation loss so far!: {self.best_validation_loss}")
                self.save_state(suffix="_BEST")
                log_entries["saved_best"] = True
            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()
            self.val_log.log(log_entries)
        # return model to training mode
        self.model.train()

    def evaluate(self, report_interval=20):
        """Evaluate the performance of the trained model on the test set."""
        log.info("Evaluating, output to directory: ", self.dump_path)
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            # Set the model to evaluation mode
            self.model.eval()
            # evaluation loop
            start_time = time()
            step_time = start_time
            steps_per_epoch = len(self.data_loaders["test"])
            for self.step, eval_data in enumerate(self.data_loaders["test"]):
                # load data
                self.data = eval_data['data'].to(self.device)
                self.target = eval_data[self.truth_key].to(self.device)
                # Run the forward procedure and output the result
                outputs, metrics = self.forward(train=False)
                # Add the local result to the final result
                if self.step == 0:
                    indices = eval_data['indices']
                    targets = self.target
                    eval_outputs = outputs
                    eval_metrics = metrics
                else:
                    indices = torch.cat((indices, eval_data['indices']))
                    targets = torch.cat((targets, self.target))
                    for k in eval_outputs.keys():
                        eval_outputs[k] = torch.cat((eval_outputs[k], outputs[k]))
                    for k in eval_metrics.keys():
                        eval_metrics[k] += metrics[k]
                # print the metrics at given intervals
                if self.rank == 0 and self.step % report_interval == 0:
                    previous_step_time = step_time
                    step_time = time()
                    average_step_time = (step_time - previous_step_time)/report_interval
                    print(f"Step {self.step}/{steps_per_epoch}"
                          f" Evaluation {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())},"
                          f" Step time {timedelta(seconds=average_step_time)},"
                          f" Total time {timedelta(seconds=step_time-start_time)}")
        for k in eval_metrics.keys():
            eval_metrics[k] /= self.step+1
        eval_outputs["indices"] = indices.to(self.device)
        eval_outputs[self.truth_key] = targets
        # Gather results from all processes
        eval_metrics = self.get_synchronized_metrics(eval_metrics)
        eval_outputs = self.get_synchronized_outputs(eval_outputs)
        if self.rank == 0:
            # Save overall evaluation results
            log.info("Saving Data...")
            for k, v in eval_outputs.items():
                np.save(self.dump_path + k + ".npy", v)
            # Compute overall evaluation metrics
            for k, v in eval_metrics.items():
                log.info(f"Average evaluation {k}: {v}")

    def save_state(self, suffix="", name=None):
        """
        Save model weights and other training state information to a file.

        Parameters
        ==========
        suffix : string
            The suffix for the filename. Should be "_BEST" for saving the best validation state.
        name : string
            The name for the filename. By default, use the engine class name followed by model class name.

        Returns
        =======
        filename : string
            Filename where the saved state is saved.
        """
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        filename = f"{self.dump_path}{name}{suffix}.pth"
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.module.state_dict()
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        log.info('Saved state as:', filename)
        return filename

    def restore_best_state(self, name=None):
        """Restore model using best model found in current directory."""
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        self.restore_state(f"{self.dump_path}{name}_BEST.pth")

    def restore_state(self, weight_file):
        """Restore model and training state from a given filename."""
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            log.info('Restoring state from', weight_file)
            # prevent loading while DDP operations are happening
            if self.is_distributed:
                torch.distributed.barrier()
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f, map_location=self.device)
            # load network weights
            self.module.load_state_dict(checkpoint['state_dict'])
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']


