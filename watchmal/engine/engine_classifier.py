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
from sys import stdout

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData

class ClassifierEngine:
    def __init__(self, model, gpu_list, dump_path):

        # create the directory for saving the log and dump files
        self.dirpath = dump_path
        print("Dump path: ", self.dirpath)

        self.model = model

        # configure the device to be used for model training and inference
        if gpu_list is not None:
            print("Requesting GPUs. GPU list : " + str(gpu_list))
            self.devids = ["cuda:{0}".format(x) for x in gpu_list]
            print("Main GPU : " + self.devids[0])
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])

                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model = DataParallel(self.model, device_ids=gpu_list, dim=0)
                
                print("CUDA is available")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, using CPU")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        
        # send model to device
        self.model.to(self.device)
        
        # Setup the parameters tp save given the model type
        if isinstance(self.model, DataParallel):
            self.model_accs = self.model.module
        else:
            self.model_accs = self.model
        
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.data_loaders = {}

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None

        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump at : {}".format(self.dirpath))
            os.makedirs(self.dirpath, exist_ok=True)
        
        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train.csv")
        self.val_log = CSVData(self.dirpath + "log_val.csv")
    
    def configure_optimizers(self, optimizer_config):
        """
        Inspired by pytorch lightning approach
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config):
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config)

    # TODO: restore old forward method
    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

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
        # TODO: fixed calls to cpu() and detach() in loss
        return {'loss'             : self.loss.cpu().detach().item(),
                'predicted_labels' : predicted_labels.cpu().numpy(),
                'softmax'          : softmax.detach().cpu().numpy(),
                'accuracy'         : accuracy,
                'raw_pred_labels'  : model_out}
    
    def backward(self):
        self.optimizer.zero_grad()  # reset accumulated gradient
        # TODO: added contiguous
        self.loss.contiguous()
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================

    def train(self, train_config):
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
        epochs          = train_config.epochs
        report_interval = train_config.report_interval
        val_interval    = train_config.val_interval
        num_val_batches = train_config.num_val_batches

        # set the iterations at which to dump the events and their metrics
        print(f"Validation Interval: {val_interval}")

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

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            times = []

            start_time = time()

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
                            val_iter = iter(self.data_loaders["validation"])

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
                                    
                    self.val_log.record(val_metrics)
                    self.val_log.write()
                    #TODO: Removed flush
                    #self.val_log.flush()

                    # Save the latest model
                    self.save_state(best=False)
                
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
                #TODO: Removed flush
                #self.train_log.flush()
                
                # print the metrics at given intervals
                if self.iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f" %
                          (self.iteration, epoch, res["loss"], res["accuracy"]))
                
                if epoch >= epochs:
                    break
        
        self.val_log.close()
        self.train_log.close()
    
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
        # TODO: this should be removed after replication
        print("evaluating in directory: ", self.dirpath)
        
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
            for it, val_data in enumerate(self.data_loaders["test"]):
                
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

                print("val_iteration : " + str(it) + " val_loss : " + str(result["loss"]) + " val_accuracy : " + str(result["accuracy"]))
                
                val_iterations += 1
                
        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        np.save(self.dirpath + "labels.npy", np.array(labels))
        np.save(self.dirpath + "predictions.npy", np.array(predictions))
        np.save(self.dirpath + "softmax.npy", np.array(softmaxes))
    
    # ========================================================================

    def restore_state(self, weight_file):
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
            self.model_accs.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
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

    # ========================================================================

    def restore_state_from_old_framework(self, weight_file):
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
            checkpoint = torch.load(f, map_location=self.device)
            
            # translation dict to convert between old names and new names
            translate_dict = {'feature_extractor':'encoder', 'classification_network':'classifier'}

            modules = list(self.model_accs._modules.keys())
            
            # verify that state dicts are the same
            old_encoder_keys = set(getattr(self.model_accs, 'feature_extractor').state_dict().keys())
            new_encoder_keys = set(checkpoint[translate_dict['feature_extractor']].keys())
            assert(old_encoder_keys == new_encoder_keys)

            old_classifier_keys = set(getattr(self.model_accs, 'classification_network').state_dict().keys())
            new_classifier_keys = set(checkpoint[translate_dict['classification_network']].keys())
            assert(old_classifier_keys == new_classifier_keys)

            for module_name in modules:
                print("Loading weights for module = ", module_name)
                module = getattr(self.model_accs, module_name)
                module.load_state_dict(checkpoint[translate_dict[module_name]])
            
        print('Restoration complete.')
    
    def replicate(self):
        """
        Overrides the validate method in Engine.py.
        
        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        """
        
        #import torch.multiprocessing
        #torch.multiprocessing.set_sharing_strategy('file_system')
        # Print start message
        num_dump_events = 3351020 #self.config.num_dump_events
        test_batch_size = 512
        
        # Setup the CSV file for logging the output, path to save the actual and reconstructed events, dataloader iterator

        self.log        = CSVData(self.dirpath+ "/test_validation_log.csv")
        np_event_path   = self.dirpath + "/test_validation_iteration_"
        data_iter       = self.data_loaders["test"]
        dump_iterations = max(1, ceil(num_dump_events/test_batch_size))
        
        print("Dump iterations = {0}".format(dump_iterations))
        save_arr_dict = {"events":[], "labels":[], "energies":[], "angles":[], "eventids":[], "rootfiles":[], "predicted_labels":[], "softmax":[]}

        #with torch.no_grad():
        self.model.eval()

        avg_loss = 0
        avg_acc = 0
        count = 0
        for iteration, data in enumerate(data_iter):

            # Extract the event data from the input data tuple
            self.data     = data['data'].float()
            self.labels   = data['labels'].long()
            #print(self.labels)
            
            self.energies = data['energies'].float()
            self.eventids = data['event_ids'].float()
            self.rootfiles = data['root_files']
            self.angles = data['angles'].float()
            
            res  = self.forward(train=False)
                
            vals = {"iteration":iteration, "loss":res["loss"], "accuracy":res["accuracy"]}
            
            # Log/Report
            self.log.record(vals)
            self.log.write()
            #TODO: Removed flush
            #self.log.flush()
            
            sys.stdout.write("val_iteration : " + str(iteration) + " val_loss : " + str(res["loss"]) + " val_accuracy : " + str(res["accuracy"]) + "\n")
            
            avg_acc += res['accuracy']
            avg_loss += res['loss']
            count += 1

            if iteration < dump_iterations:
                save_arr_dict["labels"].append(self.labels.cpu().numpy())
                save_arr_dict["energies"].append(self.energies.cpu().numpy())
                save_arr_dict["eventids"].append(self.eventids.cpu().numpy())
                save_arr_dict["rootfiles"].append(self.rootfiles)
                save_arr_dict["angles"].append(self.angles.cpu().numpy())
                save_arr_dict["predicted_labels"].append(res["predicted_labels"])
                save_arr_dict["softmax"].append(res["softmax"])
                
            elif iteration == dump_iterations:
                break
        
        print("Saving the npz dump array :")
        savez(np_event_path + "dump.npz", **save_arr_dict)
        avg_acc /= count
        avg_loss /= count
        print("Overall acc : {}, Overall loss : {}".format(avg_acc, avg_loss))
