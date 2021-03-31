"""
Class for training a fully supervised classifier
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData


class RegressionEngine:
    def __init__(self, model, rank, gpu, dump_path):
        """
        Args:
            model       ... model object that engine will use in training or evaluation
            rank        ... rank of process among all spawned processes (in multiprocessing mode)
            gpu         ... gpu that this process is running on
            dump_path   ... path to store outputs in
        """
        # create the directory for saving the log and dump files
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)

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
        self.labels = None
        self.energies = None
        self.eventids = None
        self.rootfiles = None
        self.angles = None
        self.event_ids = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = nn.MSELoss()

    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config
        Args:
            optimizer_config    ... hydra config specifying optimizer object
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config
        Args:
            data_config     ... hydra config specifying dataset
            loaders_config  ... hydra config specifying dataloaders
            is_distributed  ... boolean indicating if running in multiprocessing mode
            seed            ... seed to use to initialize dataloaders
        Parameters:
            self should have dict attribute data_loaders
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config,
                is_distributed=is_distributed, seed=seed)

    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations
        Args:
            metric_dict         ... dict containing values that are tensor outputs of a single process
        Returns:
            global_metric_dict  ... dict containing concatenated list of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, array in zip(metric_dict.keys(), metric_dict.values()):
            tensor = torch.as_tensor(array).to(self.device)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[name] = torch.cat(global_tensor)

        return global_metric_dict

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data
        Args:
            train   ... whether to compute gradients for backpropagation
        Parameters:
            self should have attributes data, labels, model, criterion, softmax
        Returns:
            dict containing loss, and model outputs
        """

        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.energies = self.energies.to(self.device)

            model_out = self.model(self.data)

            self.loss = self.criterion(model_out, self.energies)

        return {'loss': self.loss.detach().cpu().item(),
                'output': model_out.detach().cpu().numpy(),
                'raw_output': model_out}

    def backward(self):
        """
        Backward pass using the loss computed for a mini-batch
        Parameters:
            self should have attributes loss, optimizer
        """
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()  # compute new gradient
        clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()  # step params

    # ========================================================================
    # Training and evaluation loops

    def train(self, train_config):
        """
        Train the model on the training set
        Args:
            train_config    ... config specigying training parameters
        Parameters:
            self should have attributes model, data_loaders
        Outputs:
            val_log      ... csv log containing iteration, epoch, loss, accuracy for each iteration on validation set
            train_logs   ... csv logs containing iteration, epoch, loss, accuracy for each iteration on training set
        Returns: None
        """
        # initialize training params
        epochs = train_config.epochs
        report_interval = train_config.report_interval
        val_interval = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing = train_config.checkpointing
        early_stopping = train_config.early_stopping
        patience = train_config.patience

        # implement early stopping
        self.wait = 0

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()


        # initialize epoch and iteration counters
        epoch = 0.
        self.iteration = 0

        # keep track of the validation accuracy
        best_val_loss = 1.0e6

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        while (floor(epoch) < epochs):
            if self.rank == 0:
                print('Epoch', floor(epoch), 'Starting @',
                    strftime("%Y-%m-%d %H:%M:%S", localtime()))
            times = []

            start_time = time()
            iteration_time = start_time

            train_loader = self.data_loaders["train"]

            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            # local training loop for batches in a single epoch
            for i, train_data in enumerate(train_loader):

                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    # set model to eval mode
                    self.model.eval()

                    val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "saved_best": 0}

                    for val_batch in range(num_val_batches):
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            del val_iter
                            print("Fetching new validation iterator...")
                            val_iter = iter(self.data_loaders["validation"])
                            val_data = next(val_iter)

                        # extract the event data from the input data tuple
                        self.data = val_data['data'].float()
                        self.labels = val_data['labels'].long()
                        self.energies = val_data['energies'].float()
                        self.angles = val_data['angles'].float()
                        self.event_ids = val_data['event_ids'].float()

                        val_res = self.forward(False)

                        val_metrics["loss"] += val_res["loss"]

                    # return model to training mode
                    self.model.train()

                    # record the validation stats to the csv
                    val_metrics["loss"] /= num_val_batches

                    local_val_metrics = {"loss": np.array([val_metrics["loss"]])}

                    if self.is_distributed:
                        global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
                        for name, tensor in zip(global_val_metrics.keys(),
                                global_val_metrics.values()):
                            global_val_metrics[name] = np.array(tensor.cpu())
                    else:
                        global_val_metrics = local_val_metrics
                        print(type(global_val_metrics["loss"]))
                        print("not distributed")

                    if self.rank == 0:
                        # Save if this is the best model so far
                        global_val_loss = np.mean(global_val_metrics["loss"])

                        val_metrics["loss"] = global_val_loss

                        if val_metrics["loss"] < best_val_loss:
                            print('best validation loss so far!: {}'.format(best_val_loss))
                            self.save_state(best=True)
                            val_metrics["saved_best"] = 1

                            best_val_loss = val_metrics["loss"]

                        # Save the latest model if checkpointing
                        if checkpointing:
                            self.save_state(best=False)

                        self.val_log.record(val_metrics)
                        self.val_log.write()
                        self.val_log.flush()

                # Train on batch
                self.data = train_data['data'].float()
                self.labels = train_data['labels'].long()
                self.energies = train_data['energies'].float()
                self.angles = train_data['angles'].float()
                self.event_ids = train_data['event_ids'].float()

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                # Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                old_epoch = epoch
                # update the epoch and iteration
                epoch += 1. / len(train_loader)
                self.iteration += 1

                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"]}

                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()

                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print(
                        "... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                        (self.iteration, epoch, res["loss"],
                         iteration_time - start_time, iteration_time - previous_iteration_time))
                if early_stopping:
                    if (floor(epoch) - floor(old_epoch) == 1):
                        if (res["loss"] < best_val_loss):
                            self.wait = 1
                        else:
                            self.wait += 1
                            print("No improvement in validation loss")
                        print(self.wait)

                    if epoch >= epochs or self.wait > patience:
                        return
                else:
                    if epoch >= epochs:
                        break
            if early_stopping:
                if (self.wait > patience):
                    print("Training has stopped due to early stopping, Epoch %1.2f ... Best Val Loss %1.3f ... Time "
                          "Elapsed %1.3f " % (epoch, best_val_loss,
                             iteration_time - start_time))
                    break
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the test set
        Args:
            test_config ... hydra config specifying evaluation parameters
        Parameters:
            self should have attributes model, data_loaders, dirpath
        Outputs:
            indices     ... index in dataset of each event
            labels      ... actual label of each event
            predictions ... predicted label of each event
            softmax     ... softmax output over classes for each event
        Returns: None
        """
        print("evaluating in directory: ", self.dirpath)

        # Variables to output at the end
        eval_loss = 0.0
        eval_iterations = 0

        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():

            # Set the model to evaluation mode
            self.model.eval()

            # Variables for the confusion matrix
            loss, indices, energies, outputs = [], [], [], []

            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):
                # load data
                self.data = copy.deepcopy(eval_data['data'].float())
                self.energies = copy.deepcopy(eval_data['energies'].float())

                eval_indices = copy.deepcopy(eval_data['indices'].long().to("cpu"))

                # Run the forward procedure and output the result
                result = self.forward(False)

                eval_loss += result['loss']

                # Copy the tensors back to the CPU
                self.energies = self.energies.to("cpu")

                # Add the local result to the final result
                indices.extend(eval_indices)
                energies.extend(self.energies)
                outputs.extend(result['output'])

                print("eval_iteration : " + str(it) + " eval_loss : " + str(
                    result["loss"]))

                eval_iterations += 1

        # convert arrays to torch tensors
        print("loss : " + str(eval_loss / eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])

        local_eval_metrics_dict = {"eval_iterations": iterations, "eval_loss": loss}

        indices = np.array(indices)
        energies = np.array(energies)
        outputs = np.array(outputs)

        local_eval_results_dict = {"indices": indices, "energies": energies, "outputs": outputs}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)

            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(),
                        global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())

                indices = np.array(global_eval_results_dict["indices"].cpu())
                energies = np.array(global_eval_results_dict["energies"].cpu())
                outputs = np.array(global_eval_results_dict["outputs"].cpu())

        if self.rank == 0:
            print("Sorting Outputs...")
            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", sorted_indices)
            np.save(self.dirpath + "predictions.npy", outputs[sorted_indices])

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])

            print("\nAvg eval loss : " + str(val_loss / val_iterations))

    # ========================================================================
    # Saving and loading models

    def save_state(self, best=False):
        """
        Save model weights to a file.
        Args:
            best    ... if true, save as best model found, else save as checkpoint
        Outputs:
            dict containing iteration, optimizer state dict, and model state dict
        Returns: filename
        """
        filename = "{}{}{}{}".format(self.dirpath,
            str(self.model._get_name()),
            ("BEST" if best else ""),
            ".pth")

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
        """
        Restore model using best model found in current directory
        Args:
            placeholder     ... extraneous; hydra configs are not allowed to be empty
        Outputs: model params are now those loaded from best model file
        """
        best_validation_path = "{}{}{}{}".format(self.dirpath,
            str(self.model._get_name()),
            "BEST",
            ".pth")

        self.restore_state_from_file(best_validation_path)

    def restore_state(self, restore_config):
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """
        Restore model using weights stored from a previous run
        Args:
            weight_file     ... path to weights to load
        Outputs: model params are now those loaded from file
        """
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)

            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])

            # if optim is provided, load the state of the optim
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load iteration count
            self.iteration = checkpoint['global_step']

        print('Restoration complete.')