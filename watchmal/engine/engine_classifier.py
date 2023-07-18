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

from torchmetrics import AUROC, ROC

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
    """Engine for performing training or evaluation  for a classification network."""
    def __init__(self, model, rank, gpu, dump_path, label_set=None):
        """
        Parameters
        ==========
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
        self.epoch = 0.
        self.step = 0
        self.best_validation_loss = 1.0e14
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)
        self.do_early_stop=False

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}
        self.label_set = label_set

        # define the placeholder attributes
        self.data = None
        self.labels = None
        self.range = None
        self.loss = None
        self.loss_c = None
        self.loss_r = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_r = nn.HuberLoss()
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = None
        self.scheduler = None
    
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
            if self.label_set is not None:
                print(self.data_loaders[name].dataset)
                self.data_loaders[name].dataset.map_labels(self.label_set)
    
    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.
        
        Returns
        =======
        global_metric_dict : dict of torch.Tensor
            Dictionary containing concatenated list of tensor values gathered from all processes
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
            data = self.data.to(self.device)
            labels = self.labels.to(self.device)
            primary_range = self.range.to(self.device)

            model_out = self.model(data)
            
            softmax = self.softmax(model_out[0])
            pred_range = model_out[1]
            predicted_labels = torch.argmax(model_out[0], dim=-1)

            result = {'predicted_labels': predicted_labels,
                      'softmax': softmax,
                      'pred_range': pred_range,
                      'raw_pred_labels': model_out[0]}

            self.loss_c = self.criterion(model_out[0], labels)
            #print(f"True range: {primary_range}")
            #print(f"Pred range: {model_out[1]}")
            self.loss_r = self.criterion_r(model_out[1], primary_range)
            self.loss = self.loss_c + (self.loss_r/100.)
            accuracy = (predicted_labels == labels).sum().item() / float(predicted_labels.nelement())

            result['loss'] = float(self.loss.item())
            result['loss_c'] = float(self.loss_c.item())
            result['loss_r'] = float(self.loss_r.item())
            result['accuracy'] = accuracy
        
        return result
    
    def backward(self):
        """Backward pass using the loss computed for a mini-batch"""
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params

    def train(self, train_config):
        """
        Train the model on the training set.

        Parameters
        ==========
        train_config
            Hydra config specifying training parameters
        """
        # initialize training params
        epochs              = train_config.epochs
        report_interval     = train_config.report_interval
        val_interval        = train_config.val_interval
        num_val_batches     = train_config.num_val_batches
        checkpointing       = train_config.checkpointing
        early_stopping_patience      = train_config.early_stopping_patience
        save_interval = train_config.save_interval if 'save_interval' in train_config else None
        restore_best_state = train_config.restore_best_state

        if restore_best_state:
            print("PICKED RESTORE BEST STATE")
            self.restore_best_state("")

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
        self.best_validation_loss = 1.0e14

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])


        # global training loop for multiple epochs

        try:
            with self.model.join(throw_on_early_termination=True):
                self.run_epoch(epochs, report_interval, val_interval, num_val_batches, checkpointing, early_stopping_patience, save_interval, val_iter)
        except:
            if not self.is_distributed:
                print(f"Not running multi-processing: {self.rank}")
                self.run_epoch(epochs, report_interval, val_interval, num_val_batches, checkpointing, early_stopping_patience, save_interval, val_iter)
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def run_epoch(self, epochs, report_interval, val_interval, num_val_batches, checkpointing, early_stopping_patience, save_interval, val_iter):

        early_stop = False 
        for self.epoch in range(epochs):
            if self.rank == 0:
                print('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
                print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
                    
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
                    early_stop = self.validate(val_iter, num_val_batches, checkpointing, len(train_loader), early_stopping_patience)
                        
                        # Train on batch
                self.data = train_data['data']
                self.labels = train_data['labels']
                self.range = train_data['range']

                        # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                        #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                        # update the epoch and iteration
                        # self.epoch += 1. / len(self.data_loaders["train"])
                self.step += 1
                self.iteration += 1
                        
                        # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                        
                        # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                        
                        # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()

                    print("... Iteration %d ... Epoch %d ... Step %d/%d  ... Training Classification Loss %1.3f ... Training Regression Loss %1.3f ... Training Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                                (self.iteration, self.epoch+1, self.step, len(train_loader), res["loss_c"], res["loss_r"], res["accuracy"], iteration_time - start_time, iteration_time - previous_iteration_time))

                if early_stop:
                    break
                                        
            if self.scheduler is not None:
                self.scheduler.step()

            if (save_interval is not None) and ((self.epoch+1)%save_interval == 0):
                self.save_state(name=f'_epoch_{self.epoch+1}')   

            if early_stop:
                break

    def validate(self, val_iter, num_val_batches, checkpointing, iterations_per_epoch, early_stopping_patience):
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
        val_metrics = {"iteration": self.iteration, "loss": 0., "loss_c": 0., "loss_r": 0., "accuracy": 0., "saved_best": 0}
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
            self.range = val_data['range']

            val_res = self.forward(False)

            val_metrics["loss"] += val_res["loss"]
            val_metrics["loss_c"] += val_res["loss_c"]
            val_metrics["loss_r"] += val_res["loss_r"]
            val_metrics["accuracy"] += val_res["accuracy"]
        # return model to training mode
        self.model.train()
        # record the validation stats
        val_metrics["loss"] /= num_val_batches
        val_metrics["loss_c"] /= num_val_batches
        val_metrics["loss_r"] /= num_val_batches
        val_metrics["accuracy"] /= num_val_batches
        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "loss_c": np.array([val_metrics["loss_c"]]), "loss_r": np.array([val_metrics["loss_r"]]), "accuracy": np.array([val_metrics["accuracy"]])}

        if self.is_distributed:
            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                global_val_metrics[name] = np.array(tensor.cpu())
        else:
            global_val_metrics = local_val_metrics

        if self.rank == 0:
            # Save if this is the best model so far
            global_val_loss = np.mean(global_val_metrics["loss"])
            global_val_loss_c = np.mean(global_val_metrics["loss_c"])
            global_val_loss_r = np.mean(global_val_metrics["loss_r"])
            global_val_accuracy = np.mean(global_val_metrics["accuracy"])

            val_metrics["loss"] = global_val_loss
            val_metrics["loss_c"] = global_val_loss_c
            val_metrics["loss_r"] = global_val_loss_r
            val_metrics["accuracy"] = global_val_accuracy
            val_metrics["epoch"] = self.epoch

            if val_metrics["loss_c"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss_c"]
                self.best_iteration = self.iteration
                print('best validation loss so far!: {}'.format(self.best_validation_loss))
                self.save_state("BEST")
                val_metrics["saved_best"] = 1
            elif self.iteration - self.best_iteration >= int(early_stopping_patience*iterations_per_epoch):
                print("DOING EARLY STOPPING")
                self.do_early_stop=True
            print(f'CHECK early stopping: Iteration: {self.iteration}, best iteration: {self.best_iteration}, val loss: {val_metrics["loss"]}, val regression loss: {val_metrics["loss_r"]}, val classification loss: {val_metrics["loss_c"]}, best val classification loss: {self.best_validation_loss}, val acc: {val_metrics["accuracy"]}, patience: {early_stopping_patience*iterations_per_epoch}')

            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()

            self.val_log.record(val_metrics)
            self.val_log.write()
            self.val_log.flush()
        if self.do_early_stop:
            return True

    def evaluate(self, test_config):
        """Evaluate the performance of the trained model on the test set."""
        print("evaluating in directory: ", self.dirpath)

        
        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, indices, labels, predictions, softmaxes, pred_range, true_range= [],[],[],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):
                
                # load data
                self.data = eval_data['data']
                self.labels = eval_data['labels']
                self.range = eval_data['range']

                eval_indices = eval_data['indices']
                
                # Run the forward procedure and output the result
                result = self.forward(train=False)

                eval_loss += result['loss']
                eval_acc  += result['accuracy']
                
                # Add the local result to the final result
                indices.extend(eval_indices.numpy())
                labels.extend(self.labels.numpy())
                true_range.extend(self.range.numpy())
                predictions.extend(result['predicted_labels'].detach().cpu().numpy())
                softmaxes.extend(result["softmax"].detach().cpu().numpy())
                pred_range.extend(result["pred_range"].detach().cpu().numpy())
           
                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))
            
                eval_iterations += 1
        
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss, "eval_acc":accuracy}
        
        indices     = np.array(indices)
        labels      = np.array(labels)
        true_range      = np.array(true_range)
        predictions = np.array(predictions)
        softmaxes   = np.array(softmaxes)
        pred_range   = np.array(pred_range)
        
        local_eval_results_dict = {"indices":indices, "labels":labels, "true_range":true_range, "predictions":predictions, "softmaxes":softmaxes, "pred_range": pred_range}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                true_range      = np.array(global_eval_results_dict["true_range"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
                pred_range   = np.array(global_eval_results_dict["pred_range"].cpu())

        
        if self.rank == 0:
#            print("Sorting Outputs...")
#            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print(f"Saving Data to {self.dirpath}...")
            np.save(self.dirpath + "indices.npy", indices)#sorted_indices)
            np.save(self.dirpath + "labels.npy", labels)#[sorted_indices])
            np.save(self.dirpath + "predictions.npy", predictions)#[sorted_indices])
            np.save(self.dirpath + "softmax.npy", softmaxes)#[sorted_indices])
            np.save(self.dirpath + "true_range.npy", true_range)#[sorted_indices])
            np.save(self.dirpath + "pred_range.npy", pred_range)#[sorted_indices])

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))
        
    # ========================================================================
    # Saving and loading models

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
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     name,
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
        """Restore model using best model found in current directory."""
        best_validation_path = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     "BEST",
                                     ".pth")

        self.restore_state_from_file(best_validation_path)
    
    def restore_state(self, restore_config):
        """Restore model and training state from a file given in the `weight_file` entry of the config."""
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """Restore model and training state from a given filename."""
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])
            
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # load iteration count
            self.iteration = checkpoint['global_step']
        
        print('Restoration complete.')
