# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# hydra imports
from hydra.utils import instantiate

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData


class SegmentationEngine:
    def __init__(self, model, rank, gpu, dump_path):
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

        # reduction "none" so that loss can be calculated element-wise for swapped labels
        self.criterion = nn.CrossEntropyLoss(reduction = "none") 
        self.softmax = nn.Softmax(dim=1)

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)

    def get_synchronized_metrics(self, metric_dict):
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

        Parameters:
            train = whether to compute gradients for backpropagation
            self should have attributes model, criterion, softmax, data, label
        
        Returns : a dict of loss, predicted labels, softmax, accuracy, and raw model outputs
        """

        # Since height of tank is 29, add 3 empty entries to tensor to make dimension equal to 32
        # This is required to ensure max pooling of scale factor up to 8 is possible (since 32/8 is a whole integer)
        # Tensor Size out: [batch_size, 19, 32, 40]
        self.data      = torch.cat((self.data, torch.zeros(self.data.shape[0], self.data.shape[1], 3, self.data.shape[3], dtype=torch.float)), dim = 2)

        # Unsqueeze data to give it a "channels" dimension
        # Tensor Size out: [batch_size, 1, 19, 32, 40]
        self.data      = torch.unsqueeze(self.data,1)
        
        with torch.set_grad_enabled(train):

            # Move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.labels = self.labels.to(self.device)

            model_out = self.model(self.data)[:,:,:,0:29,:] #predictions are generated, only take meaningful height values (0-29)

            #Generate swapped labels between parents 2 and 3
            swappedLabels = self.swapLabels(2, 3)

            #Calculate loss
            self.loss = self.calculateLoss(model_out, swappedLabels = swappedLabels)
 
            #Calculate other metrics
            softmax          = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out,dim=1)
            accuracy = self.calculateAccuracy(predicted_labels, swappedLabels)
               
        return {'loss'             : self.loss.detach().cpu().item(),
                'predicted_labels' : predicted_labels.detach().cpu().numpy(),
                'softmax'          : softmax.detach().cpu().numpy(),
                'accuracy'         : accuracy,
                'raw_pred_labels'  : model_out}


    def swapLabels(self, label1, label2):
        """
        Swap all occurences of two distinct labels within a labels tensor

        Returns: Swapped Tensor with exact same size/dim as original tensor
        """

        #Create new instance of labels
        swappedLabels = self.labels.clone()

        #Swap labels
        parent2Mask = swappedLabels == label1
        parent3Mask = swappedLabels == label2
        swappedLabels[parent2Mask] = label2
        swappedLabels[parent3Mask] = label1

        return swappedLabels

    
    def calculateAccuracy(self, predicted_labels, swappedLabels = None):
        """
        Calculates the accuracy on current batch of data by finding the proportion of correctly identified parents
        - This means it only takes into account non-zero labels/predictions

        If swappedLabels are provided, accuracy will be calculated twice, and the event-wise maximum will be taken
        - The idea behind this is that the larger accuracy value will represent the predictions with the correct label-data parent combination
        """

        #Count correct parents
        nCorrectParents = torch.sum((self.labels == predicted_labels) & (self.labels != 0), dim = (1,2,3), dtype=float)

        #Count total parent labels (i.e. anything except 0)
        totalParents = torch.sum(self.labels != 0, dim=(1,2,3), dtype=float)

        if(swappedLabels is not None):

            #Count number of correct parents with swapped labels
            nCorrectSwappedParents = torch.sum((swappedLabels == predicted_labels) & (swappedLabels != 0), dim = (1,2,3), dtype=float)
            
            #Take element-wise maximum
            nCorrectParents = torch.max(nCorrectParents, nCorrectSwappedParents)
        
        accuracy = (torch.mean(nCorrectParents/totalParents)).item()

        return accuracy



    def calculateLoss(self, model_out, swappedLabels = None):
        """
        Calculates the loss on current batch of data using criterion function

        If swapped labels are provided, loss will be calculated twice, and the event-wise minimum will be taken
        - The idea behind this is that the smaller loss value will represent the predictions with the correct label-data parent combination
        
        If swappedLabels are not provided, only regular loss will be calculated

        Returns: Mean loss across all events in a batch
        """

        #Calculate loss with regular labelling
        loss = torch.sum(self.criterion(model_out.float(), self.labels), dim=[1,2,3])

        if(swappedLabels is not None):

            #Calculate swapped loss
            swapLoss = torch.sum(self.criterion(model_out.float(), swappedLabels), dim=[1,2,3])

            #Calculate the overall loss as the mean of the tensor of minimums from the two loss methods
            loss = torch.mean(torch.min(loss, swapLoss)) 
        
        return loss


    def backward(self):
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================

    def train(self, train_config):
        """
        Train the model on the training set.
        
        Parameters : None
        
        Outputs :
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """

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
                            val_iter = iter(self.data_loaders["validation"])
                            val_data = next(val_iter)
                        
                        # extract the event data/labels from the input data/labels tuple
                        # Tensor Size out: [batch_size, 19, 29, 40]
                        self.data      = val_data['data'].float()
                        self.labels    = val_data['segmented_labels'].long()

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

                #Measure training time to report
                startTrainTime = time()
                
                # Train on batch
                # extract the event data/labels from the input data/labels tuple
                # Tensor Size out: [batch_size, 19, 29, 40]
                self.data      = train_data['data'].float()
                self.labels    = train_data['segmented_labels'].long()

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                #print(self.data_loaders["train"])
                epoch          += 1./len(self.data_loaders["train"])
                self.iteration += 1
                
                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()

                #Calculate training time
                totalTimeElapsed = time() - startTrainTime
                timePerEpoch = totalTimeElapsed*len(self.data_loaders["train"])/60
                
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    print("Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f ... Time per Epoch %1.2f min" %
                          (self.iteration, epoch, res["loss"], res["accuracy"], timePerEpoch))
                
                if epoch >= epochs:
                    break
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the validation set.
        
        Parameters : None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
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
            loss, accuracy, indices, labels, predictions, softmaxes= [],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            
            for it, eval_data in enumerate(self.data_loaders["test"]):

                # TODO: see if copying helps
                self.data = copy.deepcopy(eval_data['data'].float())
                self.labels = copy.deepcopy(eval_data['segmented_labels'].long())

                eval_indices = copy.deepcopy(eval_data['indices'].long().to("cpu"))

                # Run the forward procedure and output the result
                result = self.forward(False)

                eval_loss += result['loss']
                eval_acc += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")

                #Plot event defined in test config file
                self.plot_event_views(eval_data, result, test_config["save_event_plots"]["startId"], test_config["save_event_plots"]["endId"], it)

                # Add the local result to the final result
                indices.extend(eval_indices)
                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])

                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))

                eval_iterations += 1
        
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss, "eval_acc":accuracy}
        
        #These are lists of np arrays
        indices     = np.array(indices)
        labels      = np.stack(labels,axis=0)
        predictions = np.stack(predictions,axis=0)
        softmaxes   = np.stack(softmaxes,axis=0)
        
        local_eval_results_dict = {"indices":indices, "labels":labels, "predictions":predictions, "softmaxes":softmaxes}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                #If distributed evaluation is ever used, these will need to be adjusted
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
        
        if self.rank == 0:
            print("Sorting Outputs...")
            sorted_indices = np.argsort(indices)
            
            # Save overall evaluation results
            # Due to the size of the output tensors, these files will be faily large
            print("Saving Data...")
            #np.save(self.dirpath + "indices.npy", sorted_indices)
            #np.save(self.dirpath + "labels.npy", labels[sorted_indices])
            #np.save(self.dirpath + "predictions.npy", predictions[sorted_indices])
            #np.save(self.dirpath + "softmax.npy", softmaxes[sorted_indices])
            
            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))


    # ========================================================================
    def plot_event_views(self, eval_data, result, startEventId, endEventId, testBatchId):
        """
        Primary plotting function on evaluation run

        Function will save plots for every specified index in the range [startEventId, endEvendId]
        - NOTE: Ensure that startEventId, endEventId are within the range [0, test_batch_size]
        - NOTE: startEventId and endEventId are both specified in test.yaml config file

        Inputs:
        - eval_data: Data to evaluate, created in evaluate function
        - result: Predictions tensor, also created in evaluate function
        - startEventId: Starting index to save plots for
        - endEventId: Ending index to save plots for
        - testBatchId: Test Batch Index

        Outputs:
        - None
        """

        for eventNumberToPlot in range(startEventId, endEventId+1):
            dirName = "outputs/Batch_" + str(testBatchId) + "_Event_" + str(eventNumberToPlot) + "/"
            os.makedirs(dirName)

            fig = plt.figure(figsize=(50,12))
            fig = self.data_loaders["test"].dataset.plot_event(fig, eval_data["data"][eventNumberToPlot], "Data", 1, cmap=plt.cm.gist_heat_r)
            fig = self.data_loaders["test"].dataset.plot_event(fig, eval_data["segmented_labels"][eventNumberToPlot], "Labels", 2, cmap=ListedColormap(["white", "gray", "yellow", "green", "red", "blue"]))
            fig = self.data_loaders["test"].dataset.plot_event(fig, result["predicted_labels"][eventNumberToPlot], "Predictions", 3, cmap=ListedColormap(["white", "gray", "yellow", "green", "red", "blue"]))

            fig.tight_layout()
            plt.savefig(dirName + "output_plot_" + str(eventNumberToPlot) + ".png")

        return

    def restore_best_state(self):
        best_validation_path = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     "BEST",
                                     ".pth")

        self.restore_state_from_file(best_validation_path)
    
    def restore_state(self, restore_config):
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """
        Restore model using weights stored from a previous run.
        
        Parameters : weight_file
        
        Outputs : 
            
        Returns : None
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
    
    def save_state(self,best=False):
        """
        Save model weights to a file.
        
        Parameters : best
        
        Outputs : 
            
        Returns : filename
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
