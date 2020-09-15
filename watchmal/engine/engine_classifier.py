# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

# hydra imports
from hydra.utils import instantiate

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys

# WatChMaL imports
from watchmal.dataset.data_module import DataModule
from watchmal.utils.logging_utils import CSVData

class ClassifierEngine:
    def __init__(self, model_config, train_config, data):
        self.model = instantiate(model_config)
        self.train_config = train_config

        # configure the device to be used for model training and inference
        if self.train_config.gpu_list is not None:
            print("Requesting GPUs. GPU list : " + str(self.train_config.gpu_list))
            self.devids = ["cuda:{0}".format(x) for x in self.train_config.gpu_list]
            print("Main GPU : " + self.devids[0])

            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model = DataParallel(self.model, device_ids=self.train_config.gpu_list, dim=0)
                print("CUDA is available")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, using CPU")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        
        # send model to device
        self.model.to(self.device)
        
        # TODO: sort out how to handle weight decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)#, weight_decay=self.train_config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # initialize dataloaders
        self.train_loader = data.train_dataloader()
        self.val_loader = data.val_dataloader()
        self.test_loader = data.test_dataloader()

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None

        # create the directory for saving the log and dump files
        self.dirpath = self.train_config.dump_path + strftime("%Y%m%d") + "/" #+ strftime("%H%M%S") + "/"

        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump at : {}".format(self.dirpath))
            os.makedirs(self.dirpath, exist_ok=True)
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train.csv")
        self.val_log = CSVData(self.dirpath + "log_val.csv")
    
    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data

        Parameters:
            train = whether to compute gradients for backpropagation
            self should have attributes model, criterion, softmax, data, label
        
        Returns : a dict of loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.labels = self.labels.to(self.device)

            model_out = self.model(self.data)

            self.loss = self.criterion(model_out, self.labels)
            
            softmax          = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out,dim=-1)
            accuracy         = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())        
            predicted_labels = predicted_labels
        
        return {'loss'             : self.loss.detach().cpu().item(),
                'predicted_labels' : predicted_labels.cpu().numpy(),
                'softmax'          : softmax.detach().cpu().numpy(),
                'accuracy'         : accuracy,
                "raw_pred_labels"  : model_out}
    
    def backward(self):
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================

    def train(self):
        """
        Train the model on the training set.
        
        Parameters : None
        
        Outputs : 
        TODO: fix training outputs
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
        print("Training...")

        # initialize training params
        epochs          = self.train_config.epochs
        report_interval = self.train_config.report_interval
        num_vals        = self.train_config.num_vals
        num_val_batches = self.train_config.num_val_batches

        # set the iterations at which to dump the events and their metrics
        dump_iterations = self.set_dump_iterations(self.train_loader)
        print(f"Validation Interval: {dump_iterations[0]}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        epoch = 0.
        self.iteration = 0

        # keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss = 1.0e6

        # initialize the iterator over the validation set
        val_iter = iter(self.val_loader)

        # global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            times = []

            start_time = time()

            # local training loop for batches in a single epoch
            for i, batch_data in enumerate(self.train_loader):

                # Using only the charge data
                self.data     = batch_data['data'].float()
                self.labels   = batch_data['labels'].long()

                self.energies = batch_data['energies'].float()
                self.angles   = batch_data['angles'].float()
                self.event_ids    = batch_data['event_ids'].float()

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                epoch          += 1./len(self.train_loader)
                self.iteration += 1

                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                
                # run validation on given intervals
                if self.iteration % dump_iterations[0] == 0:
                    # set model to eval mode
                    self.model.eval()

                    val_metrics = {"iteration": self.iteration, "epoch": epoch, "loss": 0., "accuracy": 0., "saved_best": 0}

                    for val_batch in range(num_val_batches):
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(self.val_loader)

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

                    # save if this is the best model so far
                    if val_metrics["loss"] < best_val_loss:
                        self.save_state(best=True)
                        val_metrics["saved_best"] = 1

                        best_val_loss = val_metrics["loss"]
                        print('best validation loss so far!: {}'.format(best_val_loss))
                    
                    if self.iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies", "angles", "predicted_labels", "softmax"]
                        save_arr_values = [self.data.cpu().numpy(), self.labels.cpu().numpy(), self.energies.cpu().numpy(), self.angles.cpu().numpy(), val_res["predicted_labels"], val_res["softmax"]]

                        # save the actual and reconstructed event to the disk
                        savez(self.dirpath + "/iteration_" + str(self.iteration) + ".npz",
                              **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
                    
                    self.val_log.record(val_metrics)
                    self.val_log.write()
                    self.val_log.flush()

                    # Save the latest model
                    self.save_state(best=False)
                
                # print the metrics at given intervals
                if self.iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f" %
                          (self.iteration, epoch, res["loss"], res["accuracy"]))
                
                if epoch >= epochs:
                    break
        
        self.val_log.close()
        self.train_log.close()
    
    def evaluate(self, plt_worst=0, plt_best=0):
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
        
        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes= [],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, val_data in enumerate(self.val_loader):
                
                sys.stdout.write("val_iterations : " + str(val_iterations) + "\n")
                
                self.data = val_data['data'].float()
                self.labels = val_data['labels'].long()

                # Run the forward procedure and output the result
                result = self.forward(False)

                val_loss += result['loss']
                val_acc += result['accuracy']
                
                # Copy the tensors back to the CPU
                self.labels = self.labels.to("cpu")
                
                # Add the local result to the final result
                labels.extend(self.labels)
                predictions.extend(result['predicted_labels'])
                softmaxes.extend(result["softmax"])
                
                val_iterations += 1
                
        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        np.save(self.dirpath + "labels.npy", np.array(labels))
        np.save(self.dirpath + "predictions.npy", np.array(predictions))
        np.save(self.dirpath + "softmax.npy", np.array(softmaxes))
    
    def restore_state(self, weight_file):
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
        print('Restoration complete.')
    
    # ========================================================================
    
    def save_state(self,best=False):
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename
    
    def set_dump_iterations(self, train_loader):
        """
        Determine the intervals during training at which to dump the events and metrics.
        
        Parameters :
            train_loader - Total number of validations performed throughout training
        """

        # Determine the validation interval to use depending on the total number of iterations in the current session
        valid_interval=max(1, floor(ceil(self.train_config.epochs * len(train_loader)) / self.train_config.num_vals))

        # Save the dump at the earliest validation, middle of the training and last validation near the end of training
        dump_iterations=[valid_interval, valid_interval*floor(self.train_config.num_vals/2),
                         valid_interval*self.train_config.num_vals]

        return dump_iterations
