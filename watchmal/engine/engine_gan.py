# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch import randn

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData

class GANEngine:
    def __init__(self, model, rank, gpu, dump_path):
        # create the directory for saving the log and dump files
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)

        self.data_loaders = {}

        # Loss function
        self.criterion = BCELoss()

        # Optimizers
        self.optimizerG = Adam(self.model.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.model.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

        # define the placeholder attributes
        self.data      = None
        self.labels    = None
        self.energies  = None
        self.eventids  = None
        self.rootfiles = None
        self.angles    = None
        self.event_ids = None

        self.g_loss     = None
        self.d_loss     = None
        self.dreal_loss = None
        self.dfake_loss = None

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.nz = 128
        self.fixed_noise = randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0
    
    def forward(self, train=True):
        """
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """
        # Set the correct grad_mode given the mode
        with torch.set_grad_enabled(train):
            self.data = self.data.to(self.device)
        
            # ========================================================================
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            # ========================================================================

            ###### Train with all-real batch ######

            # Format batch
            b_size = self.data.size(0)
            label = full((b_size,), self.real_label, device=self.device)

            # Forward pass real batch through D
            self.data = self.data.type(FloatTensor)

            self.model.discriminator.zero_grad()
            output = self.model.discriminator(self.data).view(-1)

            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ###### Train with all-fake batch ######

            # Generate batch of latent vectors
            noise = randn(b_size, self.nz, 1, 1, device=self.device)

            # Generate fake image batch with G
            fake = self.model.generator(noise)
            label.fill_(self.fake_label)

            # Classify all fake batch with D
            output = self.model.discriminator(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update D
            self.optimizerD.step()


            # ========================================================================
            # (2) Update Generator network: maximize log(D(G(z)))
            # ========================================================================

            self.model.generator.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.model.discriminator(fake).view(-1)

            # Calculate G's loss based on this output
            errG = self.criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            self.optimizerG.step()

            if not train:
                genimgs = self.model.generator(self.fixed_noise).cpu().detach().numpy()

            else:
                genimgs = None
            
            #del fake, output, label, noise, errD_real, errD_fake
        
        return {"g_loss"   : errG.cpu().detach().item(),
                "d_loss"   : errD.cpu().detach().item(),
                "gen_imgs" : genimgs,
                "D_x"      : D_x,
                "D_G_z1"   : D_G_z1,
                "D_G_z2"   : D_G_z2
               }
    
    def backward(self, iteration, epoch):
        """
        Backward pass using the loss computed for a mini-batch
        """
        # For the GAN, the backward pass is taken care of in the forward function
    
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
                            # TODO: still needs to be cleaned up before final push
                            time0 = time()
                            print("Fetching new validation iterator...")
                            val_iter = iter(self.data_loaders["validation"])
                            time1 = time()
                            val_data = next(val_iter)
                            time2= time()
                            print("Fetching iterator took time ", time1 - time0)
                            print("second step step took time ", time2 - time1)
                        
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
        