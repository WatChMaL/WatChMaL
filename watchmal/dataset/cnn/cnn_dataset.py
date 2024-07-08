"""
Class implementing a PMT dataset for CNNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
from torch import from_numpy, Tensor, roll, flip
import torch
import torchvision

# generic imports
import numpy as np

import random

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset, normalize
import watchmal.dataset.data_utils as du

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, MaxAbsScaler

import h5py
import joblib


class CNNDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit time and/or charge, and the second and third dimensions
    are the height and width of the CNN image. Each pixel of the image corresponds to one PMT, with PMTs arrange in an
    event-display-like format.
    """

    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None):
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
        
        self.pmt_positions = np.load(pmt_positions_file)#['pmt_image_positions']
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_positions= use_positions
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.pmt_positions[:, 0] == row) == self.data_size[1]]
        if use_positions:
            geo_file = np.load(geometry_file, 'r')
            self.geo_positions = geo_file["position"].astype(np.float32)
        #self.transforms = None 
        self.transforms = du.get_transformations(self, transforms)
        if self.transforms is None:
            self.transforms = []
        self.one_indexed = one_indexed
        self.counter=0
        if channel_scaling is None:
            channel_scaling = {}
        self.scaling = channel_scaling

        n_channels = 0
        channels = []
        if use_times:
            n_channels += 1
            channels.append('charge')
        if use_charges:
            n_channels += 1
            channels.append('time')
        if use_positions:
            n_channels += 3
            channels.append('x')
            channels.append('y')
            channels.append('z')
        if n_channels == 0:
            raise Exception("Please set 'use_times' and/or 'use_charges' to 'True' in your data config.")


        self.image_height, self.image_width = np.max(self.pmt_positions, axis=0) + 1
        # make some index expressions for different parts of the image, to use in transformations etc
        rows, counts = np.unique(self.pmt_positions[:, 0], return_counts=True)  # count occurrences of each row
        # barrel rows are those where the row appears in mpmt_positions as many times as the image width
        barrel_rows = [row for row, count in zip(rows, counts) if count == self.image_width]
        # endcap size is the number of rows before the first barrel row
        self.endcap_size = min(barrel_rows)
        self.barrel = np.s_[..., self.endcap_size:max(barrel_rows) + 1, :]
        # endcaps are assumed to be within squares centred above and below the barrel
        endcap_left = (self.image_width - self.endcap_size) // 2
        endcap_right = endcap_left + self.endcap_size
        self.top_endcap = np.s_[..., :self.endcap_size, endcap_left:endcap_right]
        self.bottom_endcap = np.s_[..., -self.endcap_size:, endcap_left:endcap_right]
       
        self.data_size = np.insert(self.data_size, 0, n_channels)



    def process_data(self, hit_pmts, hit_times, hit_charges, hit_positions=None, double_cover = None, transforms = None):
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
            if self.use_positions:
                data[2, hit_rows, hit_cols] = hit_positions[:,0]
                data[3, hit_rows, hit_cols] = hit_positions[:,1]
                data[4, hit_rows, hit_cols] = hit_positions[:,2]
        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]
        else:
            data[0, hit_rows, hit_cols] = hit_charges
            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

        return data

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)
        if self.use_positions:
            self.hit_positions = self.geo_positions[self.event_hit_pmts, :]
            hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times, "position": self.hit_positions}
        else:
            hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times}
        # apply scaling to channels
        for c, (offset, scale) in self.scaling.items():
            hit_data[c] = (hit_data[c] - offset)/scale

        
        if self.use_positions:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"], hit_positions=hit_data["position"]))
        else:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"]))
        
        # june 17 plots
        # if self.counter < 30:
        #     du.save_fig(processed_data[0], False, counter=self.counter, output_path='/data/thoriba/t2k/plots/time_plot/CNN_dead_new_parent/')
        #     du.save_fig(processed_data[1], False, counter=self.counter, output_path='/data/thoriba/t2k/plots/charge_plot/CNN_dead_new_parent/')
            
            # du.save_time_distn(hit_data['charge'], hit_data['time'], True)
            # du.save_time_distn(processed_data[0], processed_data[1], True, counter=self.counter)
        
        #self.save_fig(processed_data[0],False)
        #processed_data, displacement = self.rotate_cylinder(Tensor.numpy(processed_data))
        #self.save_fig(processed_data[0],True, displacement = displacement)
        self.counter+=1
        data_dict["data"] = processed_data
        if False:
            du.save_fig(processed_data[1],False, counter = self.counter)
        for t in self.transforms:
            #apply each transformation only half the time
            #Probably should be implemented in data_utils?
            if random.getrandbits(1):
                data_dict = t(data_dict)
        if 0:
            du.save_fig(data_dict["data"][1],True, counter = self.counter)
        processed_data = self.double_cover(data_dict["data"])
        #processed_data = du.apply_random_transformations(self.transforms, processed_data, counter = self.counter)


        return data_dict

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

        new_barrel = torch.Tensor.numpy(torch.roll(torch.from_numpy(barrel),displacement,2))
        new_top_endcap = torch.Tensor.numpy(torchvision.transforms.functional.rotate(torch.from_numpy(top_endcap), angle))
        new_bottom_endcap = torch.Tensor.numpy(torchvision.transforms.functional.rotate(torch.from_numpy(bottom_endcap), 360-angle))

        return torch.from_numpy(np.concatenate((new_top_endcap, new_barrel, new_bottom_endcap), axis=1)), displacement


    def horizontal_image_flip(self, data):
        """Perform horizontal flip of image data, permuting mPMT channels where needed."""
        return torch.flip(data[ :, :], [2])

    def vertical_image_flip(self, data):
        """Perform vertical flip of image data, permuting mPMT channels where needed."""
        return torch.flip(data[ :, :], [1])

    def rotate_image(self, data):
        """Perform 180 degree rotation of image data, permuting mPMT channels where needed."""
        return torch.flip(data[ :, :], [1, 2])

    def horizontal_flip(self, data):
        """
        Takes image-like data and returns the data after applying a horizontal flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying horizontal flip')
        return flip(data[:, :], [1])

    def vertical_flip(self, data):
        """
        Takes image-like data and returns the data after applying a vertical flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying vertical flip')
        return flip(data[:, :], [2])

    def flip_180(self, data):
        """
        Takes image-like data and returns the data after applying both a horizontal flip to the image. This is
        equivalent to a 180-degree rotation of the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('applying 180 flip')
        return self.horizontal_image_flip(self.vertical_flip(data))

    def horizontal_reflection(self, data_dict):
        """Takes CNN input data and truth info and performs horizontal flip, permuting mPMT channels where needed."""
        data_dict["data"] = self.horizontal_image_flip(data_dict["data"])
        # Note: Below assumes z-axis is the tank's azimuth axis. True for SK, HKFD; not true for IWCD & WCTE
        if "positions" in data_dict:
            data_dict["positions"][..., 1] *= -1
        if "angles" in data_dict:
            data_dict["angles"][..., 1] *= -1
        return data_dict

    def vertical_reflection(self, data_dict):
        """Takes CNN input data and truth info and performs vertical flip, permuting mPMT channels where needed."""
        data_dict["data"] = self.vertical_image_flip(data_dict["data"])
        # Note: Below assumes z-axis is the tank's azimuth axis. True for SK, HKFD; not true for IWCD & WCTE
        if "positions" in data_dict:
            data_dict["positions"][..., 2] *= -1
        if "angles" in data_dict:
            data_dict["angles"][..., 0] *= -1
            data_dict["angles"][..., 0] += np.pi
        return data_dict

    def front_back_reflection(self, data_dict):
        """
        Takes CNN input data and truth information and returns the data with horizontal flip of the left and right
        halves of the barrels and vertical flip of the end-caps. This is equivalent to reflecting the detector, swapping
        front and back. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        # Horizontal flip of the left and right halves of barrel
        left_barrel, right_barrel = np.array_split(data_dict["data"][self.barrel], 2, axis=2)
        left_barrel[:] = self.horizontal_image_flip(left_barrel)
        right_barrel[:] = self.horizontal_image_flip(right_barrel)
        # Vertical flip of the top and bottom endcaps
        data_dict["data"][self.top_endcap] = self.vertical_image_flip(data_dict["data"][self.top_endcap])
        data_dict["data"][self.bottom_endcap] = self.vertical_image_flip(data_dict["data"][self.bottom_endcap])
        # Note: Below assumes z-axis is the tank's azimuth axis. True for SK, HKFD; not true for IWCD & WCTE
        if "positions" in data_dict:
            data_dict["positions"][..., 0] *= -1
        # New azimuth angle is -(azimuth-pi) if > 0 or -(azimuth+pi) if < 0
        if "angles" in data_dict:
            data_dict["angles"][..., 1] += np.where(data_dict["angles"][..., 1] > 0, -np.pi, np.pi)
            data_dict["angles"][..., 1] *= -1
        return data_dict

    def rotation180(self, data_dict):
        """
        Takes CNN input data and truth information and returns the data with horizontal and vertical flip of the
        end-caps and shifting of the barrel rows by half the width. This is equivalent to a 180-degree rotation of the
        detector about its axis. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        # Vertical and horizontal flips of the endcaps
        data_dict["data"][self.top_endcap] = self.rotate_image(data_dict["data"][self.top_endcap])
        data_dict["data"][self.bottom_endcap] = self.rotate_image(data_dict["data"][self.bottom_endcap])
        # Roll the barrel around by half the columns
        data_dict["data"][self.barrel] = torch.roll(data_dict["data"][self.barrel], self.image_width // 2, 2)
        # Note: Below assumes y-axis is the tank's azimuth axis. True for IWCD and WCTE, not true for SK, HKFD.
        if "positions" in data_dict:
            data_dict["positions"][..., (0, 1)] *= -1
        # rotate azimuth angle by pi, keeping values in range [-pi, pi]
        if "angles" in data_dict:
            data_dict["angles"][..., 1] += np.where(data_dict["angles"][..., 1] > 0, -np.pi, np.pi)
        return data_dict

    def random_reflections(self, data_dict):
        """Takes CNN input data and truth information and randomly reflects the detector about each axis."""
        return du.apply_random_transformations([self.horizontal_reflection, self.vertical_reflection, self.rotation180], data_dict)
 
    '''
    def front_back_reflection(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal flip of the left and
        right halves of the barrels and vertical flip of the endcaps. This is equivalent to reflecting the detector
        swapping the front and back of the event-display view. The channels of the PMTs within mPMTs also have the
        appropriate permutation applied.
        """
        #print('front back reflected')
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                     # 5
        half_barrel_width = data.shape[2]//2                    # 20
        l_endcap_index = half_barrel_width - radius_endcap      # 15
        r_endcap_index = half_barrel_width + radius_endcap      # 25
        
        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :half_barrel_width]
        right_barrel = data[:, self.barrel_rows, half_barrel_width:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :half_barrel_width] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, half_barrel_width:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data

    def rotation180(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal and vertical flip of the
        endcaps and shifting of the barrel rows by half the width. This is equivalent to a 180-degree rotation of the
        detector about its axis. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        #print('rotated 180')
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25   

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:, self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data
    '''
    
    def mpmt_padding(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with part of the barrel duplicated to one
        side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels in the mPMTs permuted, to
        provide two 'views' of the detect in one image.
        """
        #print('mpmt padding')
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index+1]

        padded_data[barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data

    def double_cover(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with all parts of the detector duplicated
        and rearranged to provide a double-cover of the image, providing two 'views' of the detector from a single image
        with less blank space and physically meaningful cyclic boundary conditions at the edges of the image.

        The transformation looks something like the following, where PMTs on the end caps are numbered and PMTs on the
        barrel are letters:
        ```
                             CBALKJIHGFED
             01                01    32
             23                23    10
        ABCDEFGHIJKL   -->   DEFGHIJKLABC
        MNOPQRSTUVWX         PQRSTUVWXMNO
             45                45    76
             67                67    54
                             ONMXWVUSTRQP
        ```
        """
        #print('double cover')
        w = data.shape[2]                                                                            
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2
        half_barrel_width, quarter_barrel_width = w//2, w//4

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180 
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width
        r1_endcap_index = l1_endcap_index + 2*radius_endcap
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1:, l1_endcap_index:r1_endcap_index]
        
        padded_data[:, :barrel_row_start, l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)
        
        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]
        
        concat_order = (self.flip_180(barrel_top_half), 
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data

class CNNDatasetDeadPMT(CNNDataset):
    """
    This class does everything done by its parent 'CNNDataset'. In addition, it sets a fixed set of PMTs as 'dead' (having 0 charge and time).
    It has three additional attributes: dead_pmt_rate, dead_pmt_seed, dead_pmts.
    """

    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None, dead_pmt_rate=None, dead_pmt_seed=None):
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
        dead_pmt_rate: float
            A proportion of dead PMTs to create.
        dead_pmt_seed: int
            Seed value for randomness involving selection of dead PMTs
        """
        super().__init__(h5file, pmt_positions_file, use_times=use_times, use_charges=use_charges, use_positions=use_positions, transforms=transforms, one_indexed=one_indexed, channel_scaling=channel_scaling, geometry_file=geometry_file)
        self.dead_pmt_rate = dead_pmt_rate
        self.dead_pmt_seed = dead_pmt_seed if dead_pmt_seed is not None else 42
        
        self.set_dead_pmts()
    
    # def __getitem__(self, item):

        # data_dict = super(CNNDataset, self).__getitem__(item)

        # if self.use_positions:
        #     self.hit_positions = self.geo_positions[self.event_hit_pmts, :]
        #     hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times, "position": self.hit_positions}
        # else:
        #     hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times}
        # # apply scaling to channels
        # for c, (offset, scale) in self.scaling.items():
        #     hit_data[c] = (hit_data[c] - offset)/scale

        
        # if self.use_positions:
        #     processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"], hit_positions=hit_data["position"]))
        # else:
        #     processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"]))
        
        # if self.counter < -30:
        #     du.save_fig_dead(processed_data[1], False, self.dead_pmts, self.pmt_positions, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/charge_plot/dead_{round(self.dead_pmt_rate*100)}_dead_seed5/', dead_pmt_percent=round(self.dead_pmt_rate*100))
        #     du.save_fig_dead(processed_data[1], False, None, None, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/charge_plot/dead_{round(self.dead_pmt_rate*100)}_nodead_seed5/', dead_pmt_percent=round(self.dead_pmt_rate*100))
            
            
        #     # du.save_time_distn(hit_data['charge'], hit_data['time'], True)
        #     # du.save_time_distn(processed_data[1], processed_data[0], True, counter=self.counter, output_path='/data/thoriba/t2k/plots/time_distn/CNN_dead_5/')

        # self.counter+=1
        # data_dict["data"] = processed_data
        # if False:
        #     du.save_fig(processed_data[1],False, counter = self.counter)
        # for t in self.transforms:
        #     #apply each transformation only half the time
        #     #Probably should be implemented in data_utils?
        #     if random.getrandbits(1):
        #         data_dict = t(data_dict)
        
        # processed_data = self.double_cover(data_dict["data"])
        # return data_dict
    
    def set_dead_pmts(self):
        """
        Sets array of dead PMTs using dead_pmt_rate and dead_pmt_seed if dead_pmt_rate is not None and is in (0, 1]
        dead_pmts is an array of dead PMT IDs.
        """
        if self.dead_pmt_rate is not None and self.dead_pmt_rate > 0 and self.dead_pmt_rate <= 1:
            num_dead_pmts = min(len(self.pmt_positions), int(len(self.pmt_positions) * self.dead_pmt_rate))
            np.random.seed(self.dead_pmt_seed)
            self.dead_pmts = np.random.choice(len(self.pmt_positions), num_dead_pmts, replace=False)
            print(f'Dead PMTs were set with rate={self.dead_pmt_rate} with seed={self.dead_pmt_seed}. Here is dead PMT IDs')
            print(self.dead_pmts)
        else:
            self.dead_pmts = np.array([], dtype=int)
            print('No dead PMTs were set. If you intend to set dead PMTs, please make sure dedd PMT rate is in (0,1]')

    def process_data(self, hit_pmts, hit_times, hit_charges, hit_positions=None, double_cover = None, transforms = None):
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
        
        dead_pmts = self.dead_pmts # int Array of dead PMT IDs

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        hit_rows_d = self.pmt_positions[dead_pmts, 0]
        hit_cols_d = self.pmt_positions[dead_pmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)

        debug_mode = 0

        if self.use_times and self.use_charges:
            if debug_mode:
                print('----------------------------------')
                print('dead PMT rate', round(self.dead_pmt_rate * 100, 4), '%')
                print('Num of dead PMTs', len(dead_pmts), ' | ', round(len(dead_pmts) / len(self.pmt_positions) * 100, 4), '%', f'of {len(self.pmt_positions)} PMTs')
                print('IDs of dead PMTs', dead_pmts)
                print('Num of hit PMTs ', len(hit_pmts))
                print('Num of hit charges', len(hit_charges))
                print('Num of hit times', len(hit_times))
                
            data[0, hit_rows, hit_cols] = hit_times
            data[1, hit_rows, hit_cols] = hit_charges

            if debug_mode:
                ti_pre = np.count_nonzero(data[0])
                ch_pre = np.count_nonzero(data[1])
                print('non-zero times in data (before)', ti_pre)
                print('non-zero chrgs in data (before)', ch_pre)

            # kill
            data[0, hit_rows_d, hit_cols_d] = .0
            data[1, hit_rows_d, hit_cols_d] = .0

            if debug_mode:
                RED = '\033[91m'
                GREEN = '\033[92m'
                RESET = '\033[0m'  # Reset to default color

                ti_post = np.count_nonzero(data[0])
                ch_post = np.count_nonzero(data[1])

                print('non-zero times in data (after) ', ti_post, f'{RED}-{ti_pre - ti_post} ({round((ti_pre - ti_post)/ti_pre * 100, 4)} %) {RESET}')
                print('non-zero chrgs in data (after) ', ch_post, f'{RED}-{ch_pre - ch_post} ({round((ch_pre - ch_post)/ch_pre * 100, 4)} %) {RESET}')

            if self.use_positions:
                data[2, hit_rows, hit_cols] = hit_positions[:,0]
                data[3, hit_rows, hit_cols] = hit_positions[:,1]
                data[4, hit_rows, hit_cols] = hit_positions[:,2]

                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
                data[4, hit_rows_d, hit_cols_d] = .0

        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
            # kill
            data[0, hit_rows_d, hit_cols_d] = .0

            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

                data[1, hit_rows_d, hit_cols_d] = .0
                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
        else:
            data[0, hit_rows, hit_cols] = hit_charges
            # kill
            data[0, hit_rows_d, hit_cols_d] = .0

            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

                data[1, hit_rows_d, hit_cols_d] = .0
                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0

        return data
    


class CNNDatasetScale(CNNDataset):
    """
    A dataset class for CNN models with additional scaling functionality.
    This class extends CNNDataset to include features for scaling channel data,
    handling dead PMTs, and applying various scaling techniques to the time data.

    Attributes:
        dead_pmt_rate (float): Rate of dead PMTs to simulate.
        dead_pmt_seed (int): Seed for random number generation for dead PMTs.
        channel_scaler (dict): Configuration for scaling time channel.
        mask_train (numpy.ndarray): Boolean mask for training data.
        scaler: Scaler object or list of Scaler objects for time data transformation.
    """
    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None,
                 dead_pmt_rate=None, dead_pmt_seed=None, channel_scaler=None):
        """
        Initialize the CNNDatasetScale object.
        """
        super().__init__(h5file, pmt_positions_file, use_times=use_times, use_charges=use_charges, use_positions=use_positions, transforms=transforms, one_indexed=one_indexed, channel_scaling=channel_scaling, geometry_file=geometry_file)
        self.dead_pmt_rate = dead_pmt_rate
        self.dead_pmt_seed = dead_pmt_seed if dead_pmt_seed is not None else 42

        self.channel_scaler = channel_scaler

        self.mask_train = None

        self.set_dead_pmts()
        self.scale_time()
        # self.test_func()
    
    def get_train_mask(self):
        """
        Generate and return a boolean mask for the training data from 'dataset_index_file' in channel_scaler dict.

        Returns:
            numpy.ndarray: Boolean mask indicating training data.
        """
        if self.mask_train is not None:
            return self.mask_train

        debug_mode = 1

        if self.channel_scaler['dataset_index_file'] is not None:
                # see if possible to load the index
                train_index = np.array(np.load(self.channel_scaler['dataset_index_file'])['train_idxs'])
        else:
            train_index = None
            # think about this later
            return None
                
        if debug_mode:
            print('train index', train_index)
        
        num_events = len(self.event_hits_index)
        num_time = len(self.time)

        starts = self.event_hits_index[train_index]
        next_event_idx = np.minimum(train_index + 1, num_events - 1)
        ends = np.where(next_event_idx == len(next_event_idx), 
                        num_time,
                        self.event_hits_index[next_event_idx])
        for i in range(len(train_index)):
            if next_event_idx[i] == train_index[i] == num_events - 1:
                ends[i] = num_time
        
        mask_train = np.zeros_like(self.time, dtype=bool)
        for start, end in zip(starts, ends):
            mask_train[start:end] = True

        if debug_mode:
            print('train_idx.shape', train_index.shape)
            print('next_event_idx.shape', next_event_idx.shape)
            print('train_idx(head)', train_index[:10], train_index[10000:10010])
            print('next idx of it ', next_event_idx[:10], next_event_idx[10000:10010])
            print('starts (head)', starts[:20])
            print('ends (head)  ', ends[:20])
            print('training mask', mask_train)
            print('training mask shape', mask_train.shape)
            print('train mask L0', np.sum(mask_train))
        
        self.mask_train = mask_train
        
        return mask_train
        

    def fit_and_save_scaler(self):
        """
        Fit a scaler to the training data and save it for future use.
        The type of scaler is determined by the 'scaler_type' in channel_scaler.
        Possible choices: 'minmax', 'standard', 'robust', 'power', 'quantile_uniform', 'quantile_normal'.
        Parameters of each scaler such as range in MinMaxScaler need to be hard coded below.
        """
        # set 1 to print a bunch of info for validation
        debug_mode = 1

        if self.channel_scaler['scaler_type'] == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0.1, 1.3))
        elif self.channel_scaler['scaler_type'] == 'standard':
            self.scaler = StandardScaler()
        elif self.channel_scaler['scaler_type'] == 'robust':
            self.scaler = RobustScaler()
        elif self.channel_scaler['scaler_type'] == 'power':
            self.scaler = PowerTransformer()
        elif self.channel_scaler['scaler_type'] == 'quantile_uniform':
            self.scaler = QuantileTransformer()
        elif self.channel_scaler['scaler_type'] == 'quantile_normal':
            self.scaler = QuantileTransformer(output_distribution='normal')
    
        print('Will fit sk-learn scaler based on training set. Scaler:', self.channel_scaler['scaler_type'])

        print('Fitting the scaler on traing set. This will take some time..')
        mask_train = self.get_train_mask()
        self.scaler.fit(self.time[mask_train].reshape(-1, 1))          
        
        # saving the scaler object so that we can reuse it for another training and test (evaluation) dataset
        scaler_name_str = self.channel_scaler['scaler_type']
        joblib.dump(self.scaler, f'/data/thoriba/t2k/indices/oct20_combine_flatE/{scaler_name_str}_scaler.joblib')
        print('Scaler was fitted and saved')

        if debug_mode:
            print(self.scaler.get_params())
    
    
    def scale_time(self):
        """
        Fit and apply scaling to the time data based on the configured scaler, if transform_per_batch is False.
        If scalers are fitted, they will be saved. To fit scaler, give null for 'fitted_scaler' in channel_scaler dict.
        If no scaler information is given, no scaling is applied.
        """
        debug_mode = 1
        if self.channel_scaler is None:
            print('No scaling is done.')
            return

        if not self.initialized:
                self.initialize()

        # if scaler is provided, load the scaler, and will not re-fit the scaler(s)
        if self.channel_scaler['fitted_scaler'] is not None:
            # if type(self.channel_scaler['fitted_scaler']) == list:
            self.scaler = []
            for scaler_joblib in self.channel_scaler['fitted_scaler']:
                s = joblib.load(scaler_joblib)
                self.scaler.append(s)
                print('scaler loaded', s.get_params())
            
            if len(self.scaler) == 1:
                self.scaler = self.scaler[0]
            # else:
            #     self.scaler = joblib.load(self.channel_scaler['fitted_scaler'])
            #     print('Loaded already fitted scaler from file')
            #     print(self.scaler.get_params())
        # fit two scalers when they are chained
        elif self.channel_scaler['scaler_type'] == 'chain_fitting':
            print('Hello. This is just for fitting two different scalers. Should not be used in training/validation/testing')
            scaler_1 = QuantileTransformer(output_distribution='normal')
            # scaler_1 = RobustScaler()

            mask_train = self.get_train_mask()

            # scaler_1 = joblib.load('/home/thoriba/t2k2/t2k_ml_training/quantile_normal_scaler.joblib')
            scaler_1.fit(self.time[mask_train].reshape(-1, 1))
            # unif = np.random.uniform(low=0, high=1, size=mask_train.shape)

            # mask_sample = unif * mask_train

            # print('sample size', np.sum(mask_sample > .8))

            print(np.sum(mask_train))

            # post_transf_samples = scaler_1.transform(self.time[mask_sample > .8].reshape(-1, 1)).reshape(-1,)
            post_transf_samples = scaler_1.transform(self.time[mask_train].reshape(-1, 1)).reshape(-1,)

            print('fitting 2nd one')
            # self.time = scaler_1.transform(self.time.reshape(-1, 1)).reshape(-1,)
            scaler_2 = MinMaxScaler(feature_range=(0.1, 1.1))
            # fit on transformed version
            scaler_2.fit(post_transf_samples.reshape(-1, 1))

            post_transf_samples = scaler_2.transform(post_transf_samples.reshape(-1, 1)).reshape(-1,)
            sample = scaler_2.transform(self.time.reshape(-1, 1)).reshape(-1,)

            joblib.dump(scaler_1, f'chain_scaler_1_normal_flatE.joblib')
            joblib.dump(scaler_2, f'chain_scaler_2_minmax_for_normal_flatE.joblib')

        # test for chain scaler.
        elif self.channel_scaler['scaler_type'] == 'chain':
            print('this is for development. not to be used for training/val/test')
            scaler_1 = joblib.load('/home/thoriba/t2k2/t2k_ml_training/chain_scaler_1_quantile_normal.joblib')
            scaler_2 = joblib.load('/home/thoriba/t2k2/t2k_ml_training/chain_scaler_2_minmax.joblib')

            # scaler_1.transform(self.time[mask_sample > .8].reshape(-1, 1)).reshape(-1,)
            post_transf_samples = scaler_1.transform(self.time[:100000000].reshape(-1, 1)).reshape(-1,)
            post_transf_samples = scaler_2.transform(post_transf_samples.reshape(-1, 1)).reshape(-1,)
            du.generic_histogram(post_transf_samples, 'PMT Time [ns]', '/data/thoriba/t2k/plots/scaling_test/', f'train_hit_pmt_time_post_minmax(normal)_', y_name = None, range=None, label=None, bins=200, doNorm=True)
        else:
            print('Fitting a new scaler based on training data')
            self.fit_and_save_scaler()
            print('Finished fitting scaler')
        
        if debug_mode:
            time10_pre = self.time[:10]
            pre_str = f"self.time shape: {self.time.shape}, dtype: {self.time.dtype}"
        
        if debug_mode:
            scaler_name_str = self.channel_scaler['scaler_type']
            output_path = '/data/thoriba/t2k/plots/scaling_test/'
            # du.generic_histogram(self.time[mask_train], 'PMT Time [ns]', output_path, f'train_hit_pmt_time_pre_scaling', y_name = None, range=[0,2000], label=None, bins=200, doNorm=True)

        # apply scaling at once
        if not self.channel_scaler['transform_per_batch']:        
            print('Transforming the entire self.time (regardless of train/val/test split if such split exists) based on the scaler. This will take some time..')
            self.time = self.scaler.transform(self.time.reshape(-1, 1)).reshape(-1,)
            print('self.time was successfully scaled')
        else:
            print('Transform data on per batch basis')

        if debug_mode:
            print(pre_str)
            time10_post = self.time[:10] 
            print('before', time10_pre)
            print('after', time10_post)
            print(f"self.time shape: {self.time.shape}, dtype: {self.time.dtype}")
            # du.generic_histogram(self.time[mask_train], 'PMT Time [ns]', output_path, f'train_hit_pmt_time_post_{scaler_name_str}', y_name = None, range=None, label=None, bins=200, doNorm=True)


    def __getitem__(self, item):
        """
        This method extends the parent class's __getitem__ method to apply
        additional processing, including time scaling if configured to transform data on per-batch basis by 'transform_per_batch' == True.

        This method does everything done by parent of it, EXCEPT for applying simple linear scaler in the form of (X - offset) / scale.
        
        Returns:
            dict: A dictionary containing the processed data for the requested item.
        """

        data_dict = super(CNNDataset, self).__getitem__(item)

        if self.use_positions:
            self.hit_positions = self.geo_positions[self.event_hit_pmts, :]
            hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times, "position": self.hit_positions}
        else:
            hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times}
        
        # if self.counter < 30:
        #     if self.use_positions:
        #         processed_data_debug = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"], hit_positions=hit_data["position"]))
        #     else:
        #         processed_data_debug = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"]))
        #     du.save_fig_dead(processed_data_debug[0], False,  None, None, y_label='PMT Time', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/time_scale_pre_410_time/', dead_pmt_percent=0)
        #     du.save_fig_dead(processed_data_debug[1], False,  None, None, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/time_scale_pre_410_chrg/', dead_pmt_percent=0)

        """
        Linear scaler (X - offset) / scale is turned off.
        """
        # apply scaling to channels
        # for c, (offset, scale) in self.scaling.items():
        #     hit_data[c] = (hit_data[c] - offset)/scale

        # hit_data['time'] = (hit_data['time'] - 400)/1000
        # print('--------')
        # if item % 10 == 0:
        #     print('before scaling', hit_data['time'][:10], 'shape', hit_data['time'].shape)
        #     print('scaling...')
        
        if self.scaler is not None:
            if self.channel_scaler['transform_per_batch']:
                if isinstance(self.scaler, list):
                    for ssc in self.scaler:
                        hit_data['time'] = ssc.transform(hit_data['time'].reshape(-1, 1)).reshape(-1,)
                else:
                    hit_data['time'] = self.scaler.transform(hit_data['time'].reshape(-1, 1)).reshape(-1,)
        
        # if item % 10 == 0:
            # print('scaling done?')
            # print('after scaling', hit_data['time'][:10], 'shape', hit_data['time'].shape)
        
        if self.use_positions:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"], hit_positions=hit_data["position"]))
        else:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"]))
        
        # if self.counter < 30:
        #     du.save_fig_dead(processed_data[0], True,  None, None, y_label='PMT Time', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/time_scale_410_time/', dead_pmt_percent=0)
        #     du.save_fig_dead(processed_data[1], True,  None, None, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/time_scale_410_chrg/', dead_pmt_percent=0)
            
            # du.save_fig_dead(processed_data[1], False, self.dead_pmts, self.pmt_positions, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/charge_plot/dead_{round(self.dead_pmt_rate*100)}_dead_seed5/', dead_pmt_percent=round(self.dead_pmt_rate*100))
            # du.save_fig_dead(processed_data[1], False, None, None, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/charge_plot/dead_{round(self.dead_pmt_rate*100)}_nodead_seed5/', dead_pmt_percent=round(self.dead_pmt_rate*100))

        self.counter+=1
        data_dict["data"] = processed_data
        if False:
            du.save_fig(processed_data[1],False, counter = self.counter)
        for t in self.transforms:
            #apply each transformation only half the time
            #Probably should be implemented in data_utils?
            if random.getrandbits(1):
                data_dict = t(data_dict)
        
        processed_data = self.double_cover(data_dict["data"])
        return data_dict
    
    def set_dead_pmts(self):
        """
        Sets array of dead PMTs using dead_pmt_rate and dead_pmt_seed if dead_pmt_rate is not None and is in (0, 1]
        dead_pmts is an array of dead PMT IDs.
        """
        if self.dead_pmt_rate is not None and self.dead_pmt_rate > 0 and self.dead_pmt_rate <= 1:
            num_dead_pmts = min(len(self.pmt_positions), int(len(self.pmt_positions) * self.dead_pmt_rate))
            np.random.seed(self.dead_pmt_seed)
            self.dead_pmts = np.random.choice(len(self.pmt_positions), num_dead_pmts, replace=False)
            print(f'Dead PMTs were set with rate={self.dead_pmt_rate} with seed={self.dead_pmt_seed}. Here is dead PMT IDs')
            print(self.dead_pmts)
        else:
            self.dead_pmts = np.array([], dtype=int)
            print('No dead PMTs were set. If you intend to set dead PMTs, add dead PMT rate that is in (0,1] in yaml file')

    def process_data(self, hit_pmts, hit_times, hit_charges, hit_positions=None, double_cover = None, transforms = None):
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
        
        dead_pmts = self.dead_pmts # int Array of dead PMT IDs

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        hit_rows_d = self.pmt_positions[dead_pmts, 0]
        hit_cols_d = self.pmt_positions[dead_pmts, 1]

        # for now, idea of shifting value for unhit PMTs is not used.
        # time_of_unhit_pmts = - 10
        # should be no dead PMTs for training. 

        data = np.zeros(self.data_size, dtype=np.float32) # + time_of_unhit_pmts

        debug_mode = 0

        if self.use_times and self.use_charges:
            if debug_mode:
                print('----------------------------------')
                print('dead PMT rate', round(self.dead_pmt_rate * 100, 4), '%')
                print('Num of dead PMTs', len(dead_pmts), ' | ', round(len(dead_pmts) / len(self.pmt_positions) * 100, 4), '%', f'of {len(self.pmt_positions)} PMTs')
                print('IDs of dead PMTs', dead_pmts)
                print('Num of hit PMTs ', len(hit_pmts))
                print('Num of hit charges', len(hit_charges))
                print('Num of hit times', len(hit_times))
                
            data[0, hit_rows, hit_cols] = hit_times
            data[1, hit_rows, hit_cols] = hit_charges

            if debug_mode:
                ti_pre = np.count_nonzero(data[0])
                ch_pre = np.count_nonzero(data[1])
                print('non-zero times in data (before)', ti_pre)
                print('non-zero chrgs in data (before)', ch_pre)

            # kill
            data[0, hit_rows_d, hit_cols_d] = .0
            data[1, hit_rows_d, hit_cols_d] = .0

            if debug_mode:
                RED = '\033[91m'
                GREEN = '\033[92m'
                RESET = '\033[0m'  # Reset to default color

                ti_post = np.count_nonzero(data[0])
                ch_post = np.count_nonzero(data[1])

                print('non-zero times in data (after) ', ti_post, f'{RED}-{ti_pre - ti_post} ({round((ti_pre - ti_post)/ti_pre * 100, 4)} %) {RESET}')
                print('non-zero chrgs in data (after) ', ch_post, f'{RED}-{ch_pre - ch_post} ({round((ch_pre - ch_post)/ch_pre * 100, 4)} %) {RESET}')

            if self.use_positions:
                data[2, hit_rows, hit_cols] = hit_positions[:,0]
                data[3, hit_rows, hit_cols] = hit_positions[:,1]
                data[4, hit_rows, hit_cols] = hit_positions[:,2]

                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
                data[4, hit_rows_d, hit_cols_d] = .0

        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
            # kill
            data[0, hit_rows_d, hit_cols_d] = .0

            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

                data[1, hit_rows_d, hit_cols_d] = .0
                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
        else:
            data[0, hit_rows, hit_cols] = hit_charges
            # kill
            data[0, hit_rows_d, hit_cols_d] = .0

            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

                data[1, hit_rows_d, hit_cols_d] = .0
                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0

        return data
    
    