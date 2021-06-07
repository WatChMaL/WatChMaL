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

import torch_geometric.data as tgd

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

class ClassifierEngine:
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
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None
        self.edge_index = None
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config

        Args:
            optimizer_config    ... hydra config specifying optimizer object
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed, is_graph):
        """
        Set up data loaders from loaders config

        Args:
            data_config     ... hydra config specifying dataset
            loaders_config  ... hydra config specifying dataloaders
            is_distributed  ... boolean indicating if running in multiprocessing mode
            seed            ... seed to use to initialize dataloaders
            is_graph        ... boolean indicating if using a graph network
        
        Parameters:
            self should have dict attribute data_loaders
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed, is_graph=is_graph)
    
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

    def forward(self, train=True, return_metrics=True):
        """
        Compute predictions and metrics for a batch of data

        Args:
            train   ... whether to compute gradients for backpropagation

        Parameters:
            self should have attributes data, labels, model, criterion, softmax
        
        Returns:
            dict containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.labels = self.labels.to(self.device)

            model_out = self.model(self.data)
            
            softmax          = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out, dim=-1)

            result = { 'predicted_labels' : predicted_labels.detach().cpu().numpy(),
                      'softmax'          : softmax.detach().cpu().numpy(),
                      'raw_pred_labels'  : model_out}
            
            if return_metrics:
                self.loss = self.criterion(model_out, self.labels)
                accuracy  = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())

                result['loss'] = self.loss.detach().cpu().item()
                result['accuracy'] = accuracy
        
        return result
    
    def backward(self):
        """
        Backward pass using the loss computed for a mini-batch

        Parameters:
            self should have attributes loss, optimizer
        """
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================
    # Training and evaluation loops
    
    def train_graph(self, train_config):
        """
        Train the graph model on the training set

        Args:
            train_config    ... config specifying training parameters
        
        Parameters:
            self should have attributes model, data_loaders
        
        Outputs:
            val_log      ... csv log containing iteration, epoch, loss, accuracy for each iteration on validation set
            train_logs   ... csv logs containing iteration, epoch, loss, accuracy for each iteration on training set
            
        Returns: None
        """
        
    
    
    
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
        epochs          = train_config.epochs
        report_interval = train_config.report_interval
        val_interval    = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing   = train_config.checkpointing
        is_graph = train_config['is_graph']
        
        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        epoch = 0.
        self.iteration = 0

        # keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss = 1.0e6

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        
        '''
        *************************************
        NON-GRAPH MODELS
        *************************************
        '''
        
        if not is_graph:
                
            while (floor(epoch) < epochs):
                if self.rank == 0:
                    print('Epoch',floor(epoch), 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
        
                times = []

                start_time = time()
                iteration_time = start_time

                train_loader = self.data_loaders["train"]

                # update seeding for distributed samplers
                if self.is_distributed:
                    train_loader.sampler.set_epoch(epoch)

                # local training loop for batches in a single epoch
                for i, train_data in enumerate(self.data_loaders["train"]):

                    # run validation on given intervals
                    if self.iteration % val_interval == 0:
                        # set model to eval mode
                        self.model.eval()

                        val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "accuracy": 0., "saved_best": 0}

                        for val_batch in range(num_val_batches):
                            try:
                                val_data = next(val_iter)
                            except StopIteration:
                                del val_iter
                                print("Fetching new validation iterator...")
                                val_iter = iter(self.data_loaders["validation"])
                                val_data = next(val_iter)

                            # extract the event data from the input data tuple
                            self.data      = val_data['data'].float()
                            self.labels    = val_data['labels'].long()
                            self.energies  = val_data['energies'].float()
                            self.angles    = val_data['angles'].float()
                            self.event_ids = val_data['event_ids'].float()

                            val_res = self.forward(False)

                            val_metrics["loss"] += val_res["loss"]
                            val_metrics["accuracy"] += val_res["accuracy"]

                        # return model to training mode
                        self.model.train()

                        # record the validation stats to the csv
                        val_metrics["loss"] /= num_val_batches
                        val_metrics["accuracy"] /= num_val_batches

                        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "accuracy": np.array([val_metrics["accuracy"]])}

                        if self.is_distributed:
                            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
                            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                                global_val_metrics[name] = np.array(tensor.cpu())
                        else:
                            global_val_metrics = local_val_metrics

                        if self.rank == 0:
                            # Save if this is the best model so far
                            global_val_loss = np.mean(global_val_metrics["loss"])
                            global_val_accuracy = np.mean(global_val_metrics["accuracy"])

                            val_metrics["loss"] = global_val_loss
                            val_metrics["accuracy"] = global_val_accuracy

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
                    self.data      = train_data['data'].float()
                    self.labels    = train_data['labels'].long()
                    self.energies  = train_data['energies'].float()
                    self.angles    = train_data['angles'].float()
                    self.event_ids = train_data['event_ids'].float()

                    # Call forward: make a prediction & measure the average error using data = self.data
                    res = self.forward(True)

                    #Call backward: backpropagate error and update weights using loss = self.loss
                    self.backward()

                    # update the epoch and iteration
                    epoch          += 1./len(self.data_loaders["train"])
                    self.iteration += 1

                    # get relevant attributes of result for logging
                    train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}

                    # record the metrics for the mini-batch in the log
                    self.train_log.record(train_metrics)
                    self.train_log.write()
                    self.train_log.flush()

                    # print the metrics at given intervals
                    if self.rank == 0 and self.iteration % report_interval == 0:
                        previous_iteration_time = iteration_time
                        iteration_time = time()
                        print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                              (self.iteration, epoch, res["loss"], res["accuracy"], iteration_time - start_time, iteration_time - previous_iteration_time))

                    if epoch >= epochs:
                        break
            
            self.train_log.close()
            if self.rank == 0:
                self.val_log.close()
        
        
        '''
        *************************************
        GRAPH MODELS
        *************************************
        '''
        
        if is_graph:
            # initialize training params
            epochs          = train_config.epochs
            report_interval = train_config.report_interval
            val_interval    = train_config.val_interval
            num_val_batches = train_config.num_val_batches
            checkpointing   = train_config.checkpointing

            # set the iterations at which to dump the events and their metrics
            if self.rank == 0:
                print(f"Training... Validation Interval: {val_interval}")

            # set model to training mode
            self.model.train()

            # initialize epoch and iteration counters
            epoch = 0.
            self.iteration = 0

            # keep track of the validation accuracy
            best_val_acc = 0.0
            best_val_loss = 1.0e6

            # initialize the iterator over the validation set
            val_iter = iter(self.data_loaders["validation"])

            # global training loop for multiple epochs
            while (floor(epoch) < epochs):
                if self.rank == 0:
                    print('Epoch',floor(epoch), 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

                times = []

                start_time = time()
                iteration_time = start_time

                train_loader = self.data_loaders["train"]

                # update seeding for distributed samplers
                if self.is_distributed:
                    train_loader.sampler.set_epoch(epoch)


                # local training loop for batches in a single epoch
                for data in self.data_loaders["train"]:

                    # run validation on given intervals
                    if self.iteration % val_interval == 0:
                        # set model to eval mode
                        self.model.eval()

                        val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "accuracy": 0., "saved_best": 0}

                        for val_batch in range(num_val_batches):
                            try:
                                val_data = next(val_iter)
                            except StopIteration:
                                del val_iter
                                print("Fetching new validation iterator...")
                                val_iter = iter(self.data_loaders["validation"])
                                val_data = next(val_iter)

                            # extract the event data                   
                            self.data = val_data
                            self.labels = self.data.y

                            val_res = self.forward(False)

                            val_metrics["loss"] += val_res["loss"]
                            val_metrics["accuracy"] += val_res["accuracy"]

                        # return model to training mode
                        self.model.train()

                        # record the validation stats to the csv
                        val_metrics["loss"] /= num_val_batches
                        val_metrics["accuracy"] /= num_val_batches

                        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "accuracy": np.array([val_metrics["accuracy"]])}

                        if self.is_distributed:
                            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
                            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                                global_val_metrics[name] = np.array(tensor.cpu())
                        else:
                            global_val_metrics = local_val_metrics

                        if self.rank == 0:
                            # Save if this is the best model so far
                            global_val_loss = np.mean(global_val_metrics["loss"])
                            global_val_accuracy = np.mean(global_val_metrics["accuracy"])

                            val_metrics["loss"] = global_val_loss
                            val_metrics["accuracy"] = global_val_accuracy

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
                    self.data = data
                    self.labels = self.data.y

                    # Call forward: make a prediction & measure the average error using data = self.data
                    res = self.forward(True)

                    #Call backward: backpropagate error and update weights using loss = self.loss
                    self.backward()

                    # update the epoch and iteration
                    epoch          += 1./len(self.data_loaders["train"])
                    self.iteration += 1

                    # get relevant attributes of result for logging
                    train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}

                    # record the metrics for the mini-batch in the log
                    self.train_log.record(train_metrics)
                    self.train_log.write()
                    self.train_log.flush()

                    # print the metrics at given intervals
                    if self.rank == 0 and self.iteration % report_interval == 0:
                        previous_iteration_time = iteration_time
                        iteration_time = time()
                        print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                              (self.iteration, epoch, res["loss"], res["accuracy"], iteration_time - start_time, iteration_time - previous_iteration_time))

                    if epoch >= epochs:
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

        report_test_metrics = test_config.report_test_metrics
        
        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, indices, labels, predictions, softmaxes= [],[],[],[],[],[]
            
            '''
            *************************************
            NON-GRAPH MODELS
            *************************************
            '''
            
            if not is_graph:
                
                for it, eval_data in enumerate(self.data_loaders["test"]):
                
                # load data
                self.data = copy.deepcopy(eval_data['data'].float())
                self.labels = copy.deepcopy(eval_data['labels'].long())
                
                eval_indices = copy.deepcopy(eval_data['indices'].long().to("cpu"))
                
                # Run the forward procedure and output the result
                result = self.forward(train=False, return_metrics=report_test_metrics)

                if report_test_metrics:
                    eval_loss += result['loss']
                    eval_acc  += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")
                
                # Add the local result to the final result
                indices.extend(eval_indices)
                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])

                if report_test_metrics:
                    print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))
                else:
                    print("eval_iteration : " + str(it))

                eval_iterations += 1
            
            '''
            *************************************
            GRAPH MODELS
            *************************************
            '''
        
            elif is_graph:
                
                for data in self.data_loaders["test"]:
                        
                self.data = data
                self.labels = self.data.y
                result = self.forward(train=False, return_metrics=report_test_metrics)

                if report_test_metrics:
                    eval_loss += result['loss']
                    eval_acc  += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")

                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])

                if report_test_metrics:
                    print("eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))
                

                eval_iterations += 1
        
       
    
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss, "eval_acc":accuracy}
        
        indices     = np.array(indices)
        labels      = np.array(labels)
        predictions = np.array(predictions)
        softmaxes   = np.array(softmaxes)
        
        if not is_graph:
            
            if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
        
        if self.rank == 0:
            print("Sorting Outputs...")
            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", sorted_indices)
            np.save(self.dirpath + "labels.npy", labels[sorted_indices])
            np.save(self.dirpath + "predictions.npy", predictions[sorted_indices])
            np.save(self.dirpath + "softmax.npy", softmaxes[sorted_indices])

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))
        
        elif is_graph:
            local_eval_results_dict = {"labels":labels, "predictions":predictions, "softmaxes":softmaxes}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                
                
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
        
        if self.rank == 0:
            print("Saving Data...")
            np.save(self.dirpath + "labels.npy", labels)
            np.save(self.dirpath + "predictions.npy", predictions)
            np.save(self.dirpath + "softmax.npy", softmaxes)

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))
        

        
        
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
