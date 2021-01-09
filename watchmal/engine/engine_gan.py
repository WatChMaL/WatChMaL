# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch import full, FloatTensor
from torch import randn
from torch.nn import BCELoss
from torch.optim import Adam

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData

# generic imports
import numpy as np
from math import floor, ceil
from time import strftime, localtime, time
import os

class GANEngine:
    def __init__(self, model, rank, gpu, dump_path):
        # create the directory for saving the log and dump files
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)

        self.data_loaders = {}

        # TODO: adapt model to be distributed
        self.is_distributed = False

        # Loss function
        self.criterion = BCELoss()

        # Optimizers
        # TODO: move optimizers to configs
        lr = 0.0002
        self.optimizerG = Adam(self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.model.discriminator.parameters(), lr=0.1*lr, betas=(0.5, 0.999))

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

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.nz = 128
        self.fixed_noise = randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # set image interval
        self.img_interval = 500
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config
        """
        # TODO: rework GAN optimizers
        #self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())
        print("Optimizers in init")
    
    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)
    
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
            # TODO: better way of setting batch size
            b_size = self.data.size(0)
            # TODO: changed labels to float
            label = full((b_size,), self.real_label, dtype=torch.float).to(self.device)

            # Forward pass real batch through D
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
                #"d_loss"   : errD.cpu().detach().item(),
                "d_loss_fake"   : errD_fake.cpu().detach().item(),
                "d_loss_real"   : errD_real.cpu().detach().item(),
                "gen_imgs" : genimgs,
                "D_x"      : D_x,
                "D_G_z1"   : D_G_z1,
                "D_G_z2"   : D_G_z2
               }
    
    def backward(self):
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

        # Create directory for images generated in training
        os.mkdir(os.path.join(self.dirpath, 'imgs'))

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
                
                # Train on batch
                self.data      = train_data['data'].float()
                self.labels    = train_data['labels'].long()
                self.energies  = train_data['energies'].float()
                self.angles    = train_data['angles'].float()
                self.event_ids = train_data['event_ids'].float()

                # TODO: get running on 19 channels
                # Collapse data by summing in each mPMT
                self.data = torch.sum(self.data, dim=1, keepdim=True)

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(train=True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                epoch          += 1./len(self.data_loaders["train"])
                self.iteration += 1
                
                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": epoch, "g_loss": res["g_loss"], "d_loss_fake": res["d_loss_fake"], "d_loss_real": res["d_loss_real"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print("... Iteration %d ... Epoch %1.2f ... Generator Loss %1.3f ... Fake Discriminator Loss %1.3f ... Real Discriminator Loss %1.3f " %
                          (self.iteration, epoch, res["g_loss"], res["d_loss_fake"], res["d_loss_real"]))
                
                #Save example images
                if self.iteration % self.img_interval ==0:
                    # set model to eval mode
                    self.model.eval()
                    # TODO: sort out how to run with train=False
                    res = self.forward(train=True)
                    # set model to training mode
                    self.model.train()

                    save_arr_keys = ["gen_imgs"]
                    save_arr_values = [res["gen_imgs"]]

                    # Save the actual and reconstructed event to the disk
                    np.savez(os.path.join(self.dirpath, 'imgs') + "/iteration_" + str(self.iteration) + ".npz",
                        **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
                
                if epoch >= epochs:
                    break
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()
        