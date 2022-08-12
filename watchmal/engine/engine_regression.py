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
from math import floor, ceil, log
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy
from scipy import stats #For interquartile range

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData
from watchmal.engine.engine_base import BaseEngine


class RegressionEngine(BaseEngine):
    def __init__(self, output_type, model, rank, gpu, dump_path):
        """
        Args:
            model       ... model object that engine will use in training or evaluation
            rank        ... rank of process among all spawned processes (in multiprocessing mode)
            gpu         ... gpu that this process is running on
            dump_path   ... path to store outputs in
        """
        # create the directory for saving the log and dump files
        super().__init__(model, rank, gpu, dump_path)

        self.output_type = output_type

        self.energies = None
        self.angles = None
        self.positions = None

        # placeholders for overall median and IQR values
        self.positions_median = 0
        self.positions_overall_IQR = []
        self.energies_median = None
        self.energies_IQR = None

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
            data = self.data.to(self.device)
            energies = self.energies.to(self.device)
            positions = torch.squeeze(self.positions).to(self.device)
            model_out = self.model(data)
            if self.output_type == 'positions':
                scaled_values, scaled_model_out = self.scale_positions(positions, model_out, self.positions_overall_IQR) #Loss with scaling
            elif self.output_type == 'energies':
                scaled_values, scaled_model_out = self.fit_transform(energies, model_out, self.energies_median, self.energies_IQR)
            self.loss = self.criterion(scaled_values, scaled_model_out)

        return {'loss': self.loss.item(),
                'output': model_out}

    def scale_positions(self, data, model_out, positions_overall_IQR):
        x_positions = data[:,0]
        y_positions = data[:,1]
        z_positions = data[:,2]
        x_outputs = model_out[:,0]
        y_outputs = model_out[:,1]
        z_outputs = model_out[:,2]
        x_pos_scale, x_out_scale = self.fit_transform(x_positions, x_outputs, self.positions_median, positions_overall_IQR[0])
        y_pos_scale, y_out_scale = self.fit_transform(y_positions, y_outputs, self.positions_median, positions_overall_IQR[1])
        z_pos_scale, z_out_scale = self.fit_transform(z_positions, z_outputs, self.positions_median, positions_overall_IQR[2])

        coordinates = torch.stack([x_pos_scale, y_pos_scale, z_pos_scale], dim=1)
        model_outputs = torch.stack([x_out_scale, y_out_scale, z_out_scale], dim=1)
        return coordinates, model_outputs

    def fit_transform(self, data, model_out, median, IQR):
        positions_scaled = ((data - median) / IQR)
        outputs_scaled = ((model_out - median) / IQR) #Scaling positions and output values
        return positions_scaled, outputs_scaled #RobustScaler

    def median_and_IQR_calculation(self, data):
        data = data.to(self.device)
        median = torch.median(data)
        q = torch.tensor([0.25, 0.75]).to(self.device)
        IQR = torch.quantile(data, q, dim=0, keepdim=True).to(self.device)
        IQR = IQR[1] - IQR [0] #Subtractting the 25th quartile from the 75th quartile
        return median, IQR

    def set_positions_IQR(self, data):
        # the concat caused the dataset to be two dimensions deeper
        uwrap_data = data[:,[0][0]]
        x_positions = uwrap_data[:,0]
        y_positions = uwrap_data[:,1]
        z_positions = uwrap_data[:,2]
        
        # median for positions is calculated, but not used as '0' is used instead and declared at the start
        x_positions_median, x_positions_IQR = self.median_and_IQR_calculation(x_positions)
        y_positions_median, y_positions_IQR = self.median_and_IQR_calculation(y_positions)
        z_positions_median, z_positions_IQR = self.median_and_IQR_calculation(z_positions)
        print(f"x position median = {x_positions_median}  IQR = {x_positions_IQR}")
        print(f"y position median = {y_positions_median}  IQR = {y_positions_IQR}")
        print(f"z position median = {z_positions_median}  IQR = {z_positions_IQR}")
        
        return [x_positions_IQR, y_positions_IQR, z_positions_IQR]

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
        save_interval = train_config.save_interval if 'save_interval' in train_config else None

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        self.epoch = 0.
        self.iteration = 0
        self.step = 0
        # keep track of the validation loss
        self.best_validation_loss = 1.0e10

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # ================================================================================
        # Retrieve all data for the validation and train set
        # for calculating an overall median and IQR for scaling
        # ================================================================================
        train_iter = iter(self.data_loaders["train"])
        all_validation_and_train_data = None
        for data in val_iter:
            if all_validation_and_train_data is None:
                all_validation_and_train_data = data[self.output_type]
            else:
                all_validation_and_train_data = torch.cat((all_validation_and_train_data, data[self.output_type]), 0)

        for data in train_iter:
            if all_validation_and_train_data is None:
                all_validation_and_train_data = data[self.output_type]
            else:
                all_validation_and_train_data = torch.cat((all_validation_and_train_data, data[self.output_type]), 0)


        if self.output_type == 'positions':
            self.positions_overall_IQR = self.set_positions_IQR(all_validation_and_train_data)
        elif self.output_type == 'energies':
            self.energies_median, self.energies_IQR = self.median_and_IQR_calculation(all_validation_and_train_data)

        # global training loop for multiple epochs
        for self.epoch in range(epochs):
            if self.rank == 0:
                print('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            times = []

            start_time = time()
            iteration_time = start_time

            train_loader = self.data_loaders["train"]
            self.step = 0
            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            # local training loop for batches in a single epoch
            for self.step, train_data in enumerate(train_loader):

                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    self.validate(val_iter, num_val_batches, checkpointing)

                # Train on batch
                self.data = train_data['data']
                self.labels = train_data['labels']
                self.energies = train_data['energies']
                self.angles = train_data['angles']
                self.positions = train_data['positions']

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                # Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                # self.epoch += 1. / len(self.data_loaders["train"])
                self.step += 1
                self.iteration += 1

                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": res["loss"]}

                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()

                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print("... Iteration %d ... Epoch %d ... Step %d/%d  ... Training Loss %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                          (self.iteration, self.epoch+1, self.step, len(train_loader), res["loss"], iteration_time - start_time, iteration_time - previous_iteration_time))

            if self.scheduler is not None:
                self.scheduler.step()

            if (save_interval is not None) and ((self.epoch+1)%save_interval == 0):
                self.save_state(name=f'_epoch_{self.epoch+1}')

        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def validate(self, val_iter, num_val_batches, checkpointing):
        # set model to eval mode
        self.model.eval()
        val_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": 0., "saved_best": 0}
        for val_batch in range(num_val_batches):
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                print("Fetching new validation iterator...")
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)

            # extract the event data from the input data tuple
            self.data = val_data['data']
            self.labels = val_data['labels']
            self.energies = val_data['energies']
            self.angles = val_data['angles']
            self.positions = val_data['positions']

            val_res = self.forward(False)

            val_metrics["loss"] += val_res["loss"]

        # return model to training mode
        self.model.train()

        # record the validation stats to the csv
        val_metrics["loss"] /= num_val_batches

        local_val_metrics = {"loss": np.array([val_metrics["loss"]])}

        if self.is_distributed:
            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                global_val_metrics[name] = np.array(tensor.cpu())
        else:
            global_val_metrics = local_val_metrics

        if self.rank == 0:
            # Save if this is the best model so far
            global_val_loss = np.mean(global_val_metrics["loss"])

            val_metrics["loss"] = global_val_loss
            val_metrics["epoch"] = self.epoch

            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                print('best validation loss so far!: {}'.format(self.best_validation_loss))
                self.save_state("BEST")
                val_metrics["saved_best"] = 1

            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()

            self.val_log.record(val_metrics)
            self.val_log.write()
            self.val_log.flush()

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
            loss, indices, energies, outputs, positions = [], [], [], [], []

            # ================================================================================
            # Retrieve all data for the test set for calculating
            # an overall median and IQR for scaling
            # ================================================================================
            test_iter = iter(self.data_loaders["test"])
            all_test_data = None
            for data in test_iter:
                if all_test_data is None:
                    all_test_data = data[self.output_type]
                else:
                    all_test_data = torch.cat((all_test_data, data[self.output_type]), 0)

            if self.output_type == 'positions':
                self.positions_overall_IQR = self.set_positions_IQR(all_test_data)
            elif self.output_type == 'energies':
                self.energies_median, self.energies_IQR = self.median_and_IQR_calculation(all_test_data)

            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):

                # load data
                self.data = eval_data['data']
                self.energies = eval_data['energies']
                self.positions = eval_data['positions']

                eval_indices = eval_data['indices']

                # Run the forward procedure and output the result
                result = self.forward(train=False)

                eval_loss += result['loss']

                # Add the local result to the final result
                indices.extend(eval_indices.numpy())
                energies.extend(self.energies.numpy())
                positions.extend(self.positions.detach().cpu().numpy())
                outputs.extend(result['output'].detach().cpu().numpy())

                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]))

                eval_iterations += 1

        # convert arrays to torch tensors
        print("loss : " + str(eval_loss / eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])

        local_eval_metrics_dict = {"eval_iterations": iterations, "eval_loss": loss}

        indices = np.array(indices)
        energies = np.array(energies)
        positions = np.array(positions)
        outputs = np.array(outputs)

        local_eval_results_dict = {"indices": indices, "energies": energies, "positions": positions, "outputs": outputs}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)

            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())

                indices   = global_eval_results_dict["indices"].cpu()
                energies  = global_eval_results_dict["energies"].cpu()
                positions = global_eval_results_dict["positions"].cpu()
                outputs   = global_eval_results_dict["outputs"].cpu()

        if self.rank == 0:
#            print("Sorting Outputs...")
#            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", indices)#sorted_indices)
            np.save(self.dirpath + "energies.npy", energies)#[sorted_indices])
            np.save(self.dirpath + "positions.npy", positions)#[sorted_indices])
            np.save(self.dirpath + "predictions.npy", outputs)#[sorted_indices])

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])

            print("\nAvg eval loss : " + str(val_loss / val_iterations))
