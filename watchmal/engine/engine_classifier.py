import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate

import numpy as np
from math import floor, ceil
from time import strftime, localtime, time

from watchmal.dataset.data_module import DataModule

class ClassifierEngine:
    def __init__(self, model_config, train_config, data):
        self.model = instantiate(model_config)
        self.train_config = train_config

        # configure device
        if (train_config.device == 'gpu'):
            print("Requesting a GPU")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("CUDA is available")
            else:
                self.device=torch.device("cpu")
                print("CUDA is not available")
        else:
            print("Sticking to CPU")
            self.device=torch.device("cpu")
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # initialize dataloaders
        self.train_loader = data.train_dataloader()
        self.val_loader = data.val_dataloader()
        self.test_loader = data.test_dataloader()

        # define the placeholder attributes
        self.data     = None
        self.labels   = None
        self.energies = None
        self.eventids = None
        self.rootfiles = None
        self.angles = None
        self.index = None
    
    def forward(self, train=True):
        """
        Args: self should have attributes model, criterion, softmax, data, label
        Returns: a dictionary of loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # move the data and the labels to the GPU (if using CPU this has no effect)
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)

            model_out = self.model(self.data)
            
            # training
            self.loss = self.criterion(model_out,self.label)
            
            softmax    = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out,dim=-1)
            accuracy   = (predicted_labels == self.label).sum().item() / float(predicted_labels.nelement())        
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
        # initialize training params
        epochs          = self.train_config.epochs
        report_interval = self.train_config.report_interval
        num_vals        = self.train_config.num_vals
        num_val_batches = self.train_config.num_val_batches

        # set the iterations at which to dump the events and their metrics
        dump_iterations = self.set_dump_iterations(self.train_loader)
        print(f"Validation Interval: {dump_iterations[0]}")

        # set neural net to training mode
        self.model.train()

        # initialize epoch and iteration counters
        epoch = 0.
        iteration = 0

        # parameter to update when saving the best model
        best_loss = 1000000.

        # initialize the iterator over the validation subset
        val_iter = iter(self.val_loader)

        # global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            times = []

            start_time = time()

            # local training loop for batches in a single epoch
            for batch_data in self.train_loader:
                #print('in loop')

                # Using only the charge data
                self.data     = batch_data[0].float()
                self.labels   = batch_data[1].long()
                self.energies = batch_data[2]
                self.angles   = batch_data[3]
                self.index    = batch_data[4]

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                epoch     += 1./len(self.train_loader)
                iteration += 1

                # get relevant attributes of result for logging
                keys   = ["iteration", "epoch", "loss", "accuracy"]
                values = [iteration, epoch, res["loss"], res["accuracy"]]
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(keys, values)
                self.train_log.write()

                 # print the metrics at given intervals
                if iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f" %
                          (iteration, epoch, res["loss"], res["accuracy"]))
                
                # run validation on given intervals
                if iteration % dump_iterations[0] == 0:
                    # set model to eval mode
                    self.model.eval()

                    curr_loss = 0.
                    val_batch = 0

                    val_keys   = ["iteration", "epoch", "loss", "accuracy"]
                    val_values = []

                    for val_batch in range(num_val_batches):
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(self.val_loader)

                        # extract the event data from the input data tuple
                        self.data     = val_data[0].float()
                        self.labels   = val_data[1].long()
                        self.energies = val_data[2].float()
                        self.angles   = val_data[3].float()

                        res = self.forward(False)

                        if val_batch == 0:
                            val_values = [iteration, epoch, res["loss"], res["accuracy"]]
                        else:
                            val_values[val_keys.index("loss")] += res["loss"]
                            val_values[val_keys.index("accuracy")] += res["accuracy"]
                        
                        curr_loss += res["loss"]
                    
                    val_values[val_keys.index("loss")] /= num_val_batches
                    val_values[val_keys.index("accuracy")] /= num_val_batches

                    # record the validation stats to the csv
                    self.val_log.record(val_keys, val_values)

                    # average the loss over the validation batch
                    curr_loss = curr_loss / num_val_batches

                    # save if this is the best model so far
                    if curr_loss < best_loss:
                        self.save_state(mode="best")
                        curr_loss = best_loss
                    
                    if iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies", "angles", "predicted_labels", "softmax"]
                        save_arr_values = [self.data.cpu().numpy(), self.labels.cpu().numpy(), self.energies.cpu().numpy(), self.angles.cpu().numpy(), res["predicted_labels"], res["softmax"]]

                        # save the actual and reconstructed event to the disk
                        savez(self.dirpath + "/iteration_" + str(iteration) + ".npz",
                              **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
                    
                    self.val_log.write()

                    # Save the latest model
                    self.save_state(mode="latest")
                
                if epoch >= epochs:
                    break
            
            print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f" %
                  (iteration, epoch, res['loss'], res['accuracy']))
        
        self.val_log.close()
        self.train_log.close()
    
    def validate(self):
        """
        Test the trained model on the validation set.
        
        Parameters: None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """

        """Overrides the validate method in Engine.py.
        
        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        """
        # Print start message
        if subset == "train":
            message = "Validating model on the train set"
        elif subset == "validation":
            message = "Validating model on the validation set"
        elif subset == "test":
            message = "Validating model on the test set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None
        
        print(message)
        return -1
