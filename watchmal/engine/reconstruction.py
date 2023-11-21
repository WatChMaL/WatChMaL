"""
Class for training a fully supervised classifier
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# generic imports
import numpy as np
from time import strftime, localtime, time

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData

from abc import ABC, abstractmethod


class ReconstructionEngine(ABC):
    def __init__(self, truth_key, model, rank, gpu, dump_path):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            nn.module object that contains the full network that the engine will use in training or evaluation.
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
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)
        self.truth_key = truth_key

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}

        # define the placeholder attributes
        self.data = None
        self.target = None
        self.loss = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config)

    def configure_optimizers(self, optimizer_config):
        """Instantiate an optimizer from a hydra config."""
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_scheduler(self, scheduler_config):
        """Instantiate a scheduler from a hydra config."""
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)
        print('Successfully set up Scheduler')

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

    def get_synchronized_concatenated_tensors(self, metric_dict):
        """
        Gathers results from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_output_dict : dict of torch.Tensor
            Dictionary containing concatenated tensor values gathered from all processes
        """
        global_output_dict = {}
        for name, tensor in metric_dict.items():
            if self.rank == 0:
                global_tensor = [torch.zeros_like(tensor, device=self.device) for _ in range(self.ngpus)]
            else:
                global_tensor = None
            torch.distributed.gather(global_tensor, tensor, 0)
            if self.rank == 0:
                global_output_dict[name] = torch.cat(global_tensor).detach().cpu().numpy()
        return global_output_dict
    
    def get_synchronized_mean_metrics(self, metric_dict):
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
            torch.distributed.reduce(tensor, 0)
            if self.rank == 0:
                global_metric_dict[name] = tensor.item()/self.ngpus
        return global_metric_dict

    @abstractmethod
    def forward(self, train=True):
        pass
    
    def backward(self):
        """Backward pass using the loss computed for a mini-batch"""
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()  # compute new gradient
        self.optimizer.step()  # step params

    def train(self, train_config):
        """
        Train the model on the training set.

        Parameters
        ==========
        train_config
            Hydra config specifying training parameters
        """
        # initialize training params
        epochs = train_config.epochs
        report_interval = train_config.report_interval
        val_interval = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing = train_config.checkpointing
        save_interval = train_config.save_interval if 'save_interval' in train_config else None

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        # keep track of the validation loss
        self.best_validation_loss = 1.0e10

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        for self.epoch in range(epochs):
            if self.rank == 0:
                print('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            start_time = time()
            iteration_time = start_time

            train_loader = self.data_loaders["train"]
            self.step = 0
            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            # local training loop for batches in a single epoch
            steps_per_epoch = len(train_loader)
            for self.step, train_data in enumerate(train_loader):

                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    self.validate(val_iter, num_val_batches, checkpointing)

                # Train on batch
                self.data = train_data['data'].to(self.device)
                self.target = train_data[self.truth_key].to(self.device)

                # Call forward: make a prediction & measure the average error using data = self.data
                outputs, metrics = self.forward(True)

                # Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                # self.epoch += 1. / len(self.data_loaders["train"])
                self.step += 1
                self.iteration += 1

                # get relevant attributes of result for logging
                log_entries = {"iteration": self.iteration, "epoch": self.epoch, **metrics}

                # record the metrics for the mini-batch in the log
                self.train_log.record(log_entries)
                self.train_log.write()
                self.train_log.flush()

                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print(f"Iteration {self.iteration}, Epoch {self.epoch+1}/{epochs}, Step {self.step}/{steps_per_epoch}"
                          f" Training {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())}, Time Elapsed"
                          f" {iteration_time - start_time:.1f}s, Iteration Time {iteration_time - previous_iteration_time:.1f}s")

            if self.scheduler is not None:
                self.scheduler.step()

            if self.rank == 0 and (save_interval is not None) and ((self.epoch+1) % save_interval == 0):
                self.save_state(name=f'_epoch_{self.epoch+1}')

        self.train_log.close()
        if self.rank == 0:
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
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                print("Fetching new validation iterator...")
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)

            # extract the event data from the input data tuple
            self.data = val_data['data'].to(self.device)
            self.target = val_data[self.truth_key].to(self.device)

            outputs, metrics = self.forward(False)

            if val_metrics is None:
                val_metrics = metrics
            else:
                for k, v in metrics.items():
                    val_metrics[k] += v

        # record the validation stats to the csv
        val_metrics = {k: v/num_val_batches for k, v in val_metrics.items()}

        if self.is_distributed:
            val_metrics = self.get_synchronized_mean_metrics(val_metrics)
        else:
            val_metrics = {k: v.item() for k, v in val_metrics.items()}

        if self.rank == 0:
            log_entries = {"Iteration": self.iteration, "epoch": self.epoch, **val_metrics, "saved_best": False}
            print(f"  Validation {', '.join(f'{k}: {v:.5g}' for k, v in val_metrics.items())}")
            # Save if this is the best model so far
            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                print(f"  Best validation loss so far!: {self.best_validation_loss}")
                self.save_state("BEST")
                log_entries["saved_best"] = True
            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()
            self.val_log.record(log_entries)
            self.val_log.write()
            self.val_log.flush()

        # return model to training mode
        self.model.train()

    def evaluate(self, test_config):
        """Evaluate the performance of the trained model on the test set."""
        print("evaluating in directory: ", self.dirpath)

        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():

            # Set the model to evaluation mode
            self.model.eval()

            # Variables for the outputs
            self.data = next(iter(self.data_loaders["test"]))['data']
            self.target = next(iter(self.data_loaders["test"].dataset))[self.truth_key]
            # A forward run just to figure out the outputs
            outputs, metrics = self.forward(train=False)
            indices = torch.zeros((0,), device=self.device)
            targets = torch.zeros((0, *self.target.shape), device=self.device)
            eval_outputs = {k: torch.zeros((0, *v[0].shape), device=self.device) for k, v in outputs.items()}
            eval_metrics = {k: torch.zeros((0, *v[0].shape), device=self.device) for k, v in metrics.items()}
            # evaluation loop
            start_time = time()
            iteration_time = start_time
            steps_per_epoch = len(self.data_loaders["test"])
            for self.step, eval_data in enumerate(self.data_loaders["test"]):
                # load data
                self.data = eval_data['data'].to(self.device)
                self.target = eval_data[self.truth_key].squeeze().to(self.device)
                # Run the forward procedure and output the result
                outputs, metrics = self.forward(train=False)
                # Add the local result to the final result
                indices = torch.cat((indices, eval_data['indices']))
                targets = torch.cat((targets, self.target))
                for k in eval_outputs.keys():
                    eval_outputs[k] = torch.cat((eval_outputs[k], outputs[k]))
                for k in eval_metrics.keys():
                    eval_metrics[k] = torch.cat((eval_metrics[k], metrics[k]))
                # print the metrics at given intervals
                if self.rank == 0 and self.step % test_config.report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print(f"Step {self.step}/{steps_per_epoch}"
                          f" Evaluation {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())}, Time Elapsed"
                          f" {iteration_time - start_time:.1f}s, Iteration Time {iteration_time - previous_iteration_time:.1f}s")
        eval_outputs["indices"] = indices
        eval_outputs["targets"] = targets
        if self.is_distributed:
            # Gather results from all processes
            eval_metrics = self.get_synchronized_concatenated_tensors(eval_metrics)
            eval_outputs = self.get_synchronized_concatenated_tensors(eval_outputs)
        else:
            eval_metrics = {k: v.detach().cpu().numpy() for k, v in eval_metrics.items()}
            eval_outputs = {k: v.detach().cpu().numpy() for k, v in eval_outputs.items()}
        if self.rank == 0:
            # Save overall evaluation results
            print("Saving Data...")
            for k, v in eval_outputs.items():
                np.save(self.dirpath + k + ".npy", v)
            # Compute overall evaluation metrics
            for k, v in eval_metrics.items():
                print(f"Average evaluation {k}: {np.mean(v)}")

    def save_state(self, name=""):
        """
        Save model weights and other training state information to a file.

        Parameters
        ==========
        name
            Suffix for the filename. Should be "BEST" for saving the best validation state.

        Returns
        =======
        filename : string
            Filename where the saved state is saved.
        """

        if self.is_distributed and self.rank != 0:
            print("Attempted to save state, but not rank 0! NOT saving state!")
            return

        filename = f"{self.dirpath}{str(self.model._get_name())}{name}.pth"

        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.model_accs.state_dict()

        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_best_state(self, placeholder):
        """Restore model using best model found in current directory."""
        self.restore_state_from_file(f"{self.dirpath}{str(self.model._get_name())}BEST.pth")

    def restore_state(self, restore_config):
        """Restore model and training state from a file given in the `weight_file` entry of the config."""
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """Restore model and training state from a given filename."""
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # prevent loading while DDP operations are happening
            if self.is_distributed:
                torch.distributed.barrier()
            # torch interprets the file, then we can access using string keys
            if torch.cuda.is_available():
                checkpoint = torch.load(f)
            else:
                checkpoint = torch.load(f, map_location=torch.device('cpu'))

            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])

            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load iteration count
            self.iteration = checkpoint['global_step']

        print('Restoration complete.')


class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, label_set=None):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            nn.module object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path)

        self.softmax = torch.nn.Softmax(dim=1)
        self.label_set = label_set

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
        super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
        if self.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            model_out = self.model(self.data)
            softmax = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out, dim=-1)
            self.loss = self.criterion(model_out, self.target)
            accuracy = (predicted_labels == self.target).sum() / float(predicted_labels.nelement())
            outputs = {'softmax': softmax}
            metrics = {'loss': self.loss,
                       'accuracy': accuracy}
        return outputs, metrics


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, output_center=0, output_scale=1):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            nn.module object that contains the full network that the engine will use in training or evaluation.
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

        self.target = None

        # placeholders for overall median and IQR values
        self.positions_median = 0
        self.positions_overall_IQR = []
        self.energies_median = None
        self.energies_IQR = None

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
            scaled_target = self.scale_values(self.target)
            scaled_model_out = self.scale_values(model_out)
            self.loss = self.criterion(scaled_model_out, scaled_target)
            outputs = {self.truth_key: model_out}
            metrics = {'loss': self.loss.item()}
        return outputs, metrics

    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled