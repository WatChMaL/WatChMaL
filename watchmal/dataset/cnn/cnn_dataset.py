"""
Class implementing a PMT dataset for CNNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
from torch import from_numpy, Tensor, roll
import torchvision

# generic imports
import numpy as np

# WatChMaL imports
from WatChMaL.watchmal.dataset.h5_dataset import H5Dataset
import WatChMaL.watchmal.dataset.data_utils as du

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse


class CNNDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit time and/or charge, and the second and third dimensions
    are the height and width of the CNN image. Each pixel of the image corresponds to one PMT, with PMTs arrange in an
    event-display-like format.
    """

    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, transforms=None, one_indexed=False):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge and/or time
        data is formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to
        one PMT and the channels correspond to charge and/or time at each PMT. The PMTs are placed in the image
        according to a mapping provided by the numpy array in the `pmt_positions_file`.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        pmt_positions_file: string
            Location of an npz file containing the mapping from PMT IDs to CNN image pixel locations
        use_times: bool
            Whether to use PMT hit times as one of the initial CNN image channels. True by default.
        use_charges: bool
            Whether to use PMT hit charges as one of the initial CNN image channels. True by default.
        transforms
            List of random transforms to apply to data before passing to CNN for data augmentation. Currently unused for
            this dataset.
        one_indexed: bool
            Whether the PMT IDs in the H5 file are indexed starting at 1 (like SK tube numbers) or 0 (like WCSim PMT
            indexes). By default, zero-indexing is assumed.
        """
        super().__init__(h5file)
        
        self.pmt_positions = np.load(pmt_positions_file)['pmt_image_positions']
        self.use_times = use_times
        self.use_charges = use_charges
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.pmt_positions[:, 0] == row) == self.data_size[1]]
        self.transforms = None 
        #self.transforms = du.get_transformations(transformations, transforms)
        self.one_indexed = one_indexed
        self.counter=0

        n_channels = 0
        if use_times:
            n_channels += 1
        if use_charges:
            n_channels += 1
        if n_channels == 0:
            raise Exception("Please set 'use_times' and/or 'use_charges' to 'True' in your data config.")
       
        self.data_size = np.insert(self.data_size, 0, n_channels)

    def rotate_cylinder(self, data):


        hit_pmts = self.event_hit_pmts
        if self.one_indexed:
            hit_pmts = hit_pmts-1  # SK cable numbers start at 1

        num_cols = data.shape[1]

        min_barrel_row = np.amin(self.barrel_rows)
        max_barrel_row = np.amax(self.barrel_rows)

        top_endcap = data[:,0:min_barrel_row,:]
        bottom_endcap = data[:,max_barrel_row+1:,:]
        barrel = data[:,min_barrel_row:max_barrel_row+1,:]

        rng = np.random.default_rng()
        displacement = rng.integers(0,high=num_cols)
        angle = 360*(displacement/num_cols)

        new_barrel = Tensor.numpy(roll(from_numpy(barrel),displacement,2))
        new_top_endcap = Tensor.numpy(torchvision.transforms.functional.rotate(from_numpy(top_endcap), angle))
        new_bottom_endcap = Tensor.numpy(torchvision.transforms.functional.rotate(from_numpy(bottom_endcap), 360-angle))

        return from_numpy(np.concatenate((new_top_endcap, new_barrel, new_bottom_endcap), axis=1)), displacement


    def process_data(self, hit_pmts, hit_times, hit_charges):
        """
        Returns event data from dataset associated with a specific index

        Parameters
        ----------
        hit_pmts: array_like of int
            Array of hit PMT IDs
        hit_times: array_like of float
            Array of PMT hit times
        hit_charges: array_like of float
            Array of PMT hit charges
        
        Returns
        -------
        data: ndarray
            Array in image-like format (channels, rows, columns) for input to CNN network.
        """
        if self.one_indexed:
            hit_pmts = hit_pmts-1  # SK cable numbers start at 1

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]
        #print(f'pmt positions: {self.pmt_positions}')
        #print(f'hit pmts: {hit_pmts}')
        #print(f'hit rows: {hit_rows}')
        #print(f'hit columns: {hit_cols}')
        #print(f'barrel rows: {self.barrel_rows}')

        data = np.zeros(self.data_size, dtype=np.float32)

        if self.use_times and self.use_charges:
            data[0, hit_rows, hit_cols] = hit_times
            data[1, hit_rows, hit_cols] = hit_charges
        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
        else:
            data[0, hit_rows, hit_cols] = hit_charges

        return data

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)

        processed_data = from_numpy(self.process_data(self.event_hit_pmts, self.event_hit_times, self.event_hit_charges))
        #self.save_fig(processed_data[0],False)
        processed_data, displacement = self.rotate_cylinder(Tensor.numpy(processed_data))
        #self.save_fig(processed_data[0],True, displacement = displacement)
        self.counter+=1
        processed_data = du.apply_random_transformations(self.transforms, processed_data)

        data_dict["data"] = processed_data

        return data_dict

    def save_fig(self,data,isPost, displacement=0):
        plt.imshow(data)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("PMT Charge", rotation=270)
        plt.xlabel('X pixels')
        plt.ylabel('Y pixels')
        if isPost:
            plt.savefig('/home/fcormier/t2k/ml/t2k_ml_training/plots/'+str(self.counter)+'_post_rot_img_dis'+str(displacement)+'.png')
        else:
            plt.savefig('/home/fcormier/t2k/ml/t2k_ml_training/plots/'+str(self.counter)+'_pre_rot_img'+'.png')
        plt.clf()
