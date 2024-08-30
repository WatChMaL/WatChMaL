"""
Class implementing a PMT dataset for CNNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
import os
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
import re


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

        print('CNN: data_size', self.data_size)
        print('CNN: data_size.shape', self.data_size.shape)



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
        
        # if self.counter <= 10:
        #     du.save_fig_dead(processed_data[0], True,  None, None, y_label='PMT Time', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/dead_test/time_CNN/', dead_pmt_percent=-1)
        #     du.save_fig_dead(processed_data[1], True,  None, None, y_label='PMT Charge', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/dead_test/charge_CNN/', dead_pmt_percent=-1)
        #     du.save_fig_dead(processed_data[2], True,  None, None, y_label='PMT Dead (1)', counter=self.counter, output_path=f'/data/thoriba/t2k/plots/dead_test/dead_mask_withoutdead_CNN/', dead_pmt_percent=-1)
        

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

    dead_pmts: 1d numpy array of integers. Zero-indexed. Set in set_dead_pmts()

    """

    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None,
                 dead_pmt_rate=None, dead_pmt_seed=None, dead_pmts_file=None, use_dead_pmt_mask=False, use_hit_mask=False):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge and/or time
        data is formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to
        one PMT and the channels correspond to charge and/or time at each PMT. The PMTs are placed in the image
        according to a mapping provided by the numpy array in the `pmt_positions_file`.

        To set dead PMTs from list of dead PMT IDs, provide dead_pmts_file. Other dead PMT related parameters will be ignored.
        To set dead PMTs randomly, do not provide file location, and provide dead_pmt_rate and dead_pmt_seed.

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
            A proportion of dead PMTs to create. Within 0 and 1.
        dead_pmt_seed: int
            Seed value for randomness involving selection of dead PMTs.
        dead_pmts_file: string
            Location of a txt file containing dead PMT IDs. Each row is an ID of dead PMT.
        use_dead_pmt_mask: bool
            Whether to use PMT hit times as one of the initial CNN image channels. False by default.
        """
        super().__init__(h5file, pmt_positions_file, use_times=use_times, use_charges=use_charges, use_positions=use_positions, transforms=transforms, one_indexed=one_indexed, channel_scaling=channel_scaling, geometry_file=geometry_file)
        self.use_dead_pmt_mask = use_dead_pmt_mask
        self.dead_pmt_rate = dead_pmt_rate
        self.dead_pmt_seed = dead_pmt_seed if dead_pmt_seed is not None else 42
        self.dead_pmts_file = dead_pmts_file

        self.use_hit_mask = use_hit_mask

        if self.use_dead_pmt_mask:
            self.data_size[0] = self.data_size[0] + 1
        if self.use_hit_mask:
            self.data_size[0] = self.data_size[0] + 1
        
        print('CNNdead: data_size', self.data_size)
        print('CNNdead: data_size.shape', self.data_size.shape)
        
        self.set_dead_pmts()
    
    def set_dead_pmts(self):
        """
        Sets array of dead PMTs randomly or non-randomly depending on inputs from yaml. Zero-indexed by default.
        For random setting, sets dead PMT ID list using dead_pmt_rate and dead_pmt_seed if dead_pmt_rate is not None and is in (0, 1]
        For fixed selection, read it from .txt file, in which each row is ID of dead PMT.
        Sets:
        dead_pmts: a numpy 1-d array of integers; each element represents ID of dead PMT.
        """
        if self.dead_pmts_file is not None:
            self.dead_pmts = np.loadtxt(self.dead_pmts_file, dtype=int)
            if self.one_indexed:
                self.dead_pmts = self.dead_pmts - 1
            print(f'Dead PMTs were set non-randomly from file ({self.dead_pmts_file}). Here is dead PMT IDs (zero-indexed).')
            print(self.dead_pmts)
        elif self.dead_pmt_rate is not None and self.dead_pmt_rate > 0 and self.dead_pmt_rate <= 1:
            num_dead_pmts = min(len(self.pmt_positions), int(len(self.pmt_positions) * self.dead_pmt_rate))
            np.random.seed(self.dead_pmt_seed)
            self.dead_pmts = np.random.choice(len(self.pmt_positions), num_dead_pmts, replace=False)
            print(f'Dead PMTs were set with rate={self.dead_pmt_rate} with seed={self.dead_pmt_seed}. Here is dead PMT IDs')
            print(self.dead_pmts)
        else:
            self.dead_pmts = np.array([], dtype=int)
            print('No dead PMTs were set. If you intend to set dead PMTs, please provide dead_pmts_file for fixed dead PMTs or dead_pmt_rate for random selection')

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
        
        dead_pmts = self.dead_pmts # int array of dead PMT IDs. Zero-idxed.

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        hit_rows_d = self.pmt_positions[dead_pmts, 0]
        hit_cols_d = self.pmt_positions[dead_pmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)

        # set True to print out some information per batch for test / debug
        debug_mode = 0

        channel = 0

        if self.use_times and self.use_charges:
            if debug_mode:
                print('----------------------------------')
                if self.dead_pmt_rate is not None:
                    print('dead PMT rate', round(self.dead_pmt_rate * 100, 4), '%')
                print('Num of dead PMTs', len(dead_pmts), ' | ', round(len(dead_pmts) / len(self.pmt_positions) * 100, 4), '%', f'of {len(self.pmt_positions)} PMTs')
                print('IDs of dead PMTs', dead_pmts)
                print('Num of hit PMTs ', len(hit_pmts))
                print('Num of hit charges', len(hit_charges))
                print('Num of hit times', len(hit_times))
                
            data[0, hit_rows, hit_cols] = hit_times
            data[1, hit_rows, hit_cols] = hit_charges
            channel += 2

            if debug_mode:
                ti_pre = np.count_nonzero(data[0])
                ch_pre = np.count_nonzero(data[1])
                print('non-zero times in data (before)', ti_pre)
                print('non-zero chrgs in data (before)', ch_pre)

            # kill dead PMTs according to dead PMT IDs
            data[0, hit_rows_d, hit_cols_d] = .0
            data[1, hit_rows_d, hit_cols_d] = .0

            
            if self.use_positions:
                data[2, hit_rows, hit_cols] = hit_positions[:,0]
                data[3, hit_rows, hit_cols] = hit_positions[:,1]
                data[4, hit_rows, hit_cols] = hit_positions[:,2]

                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
                data[4, hit_rows_d, hit_cols_d] = .0

                channel += 3

            if self.use_dead_pmt_mask:
                if self.use_positions:
                    data[5, hit_rows_d, hit_cols_d] = 1
                else:
                    data[2, hit_rows_d, hit_cols_d] = 1
                channel += 1
            
            if self.use_hit_mask:
                data[channel, hit_rows, hit_cols] = 1
                data[channel, hit_rows_d, hit_cols_d] = 0
                channel += 1
            
            if debug_mode:
                RED = '\033[91m'
                GREEN = '\033[92m'
                RESET = '\033[0m'  # Reset to default color

                ti_post = np.count_nonzero(data[0])
                ch_post = np.count_nonzero(data[1])
                dm_post = np.count_nonzero(data[2])

                print('non-zero times in data (after) ', ti_post, f'{RED}-{ti_pre - ti_post} ({round((ti_pre - ti_post)/ti_pre * 100, 4)} %) {RESET}')
                print('non-zero chrgs in data (after) ', ch_post, f'{RED}-{ch_pre - ch_post} ({round((ch_pre - ch_post)/ch_pre * 100, 4)} %) {RESET}')
                print('non-zero dead pmt mask in data ', dm_post)
                print('hit_rows_dead',np.sort(hit_rows_d))
                print('idx of non-zero in mask', np.sort(np.where(data[2] == 1)[0]))

                print('hit_cols_dead',np.sort(hit_cols_d))
                print('idx of non-zero in mask', np.sort(np.where(data[2] == 1)[1]))

        elif self.use_times:
            data[0, hit_rows, hit_cols] = hit_times
            # kill
            data[0, hit_rows_d, hit_cols_d] = .0

            if self.use_positions:
                data[1, hit_rows, hit_cols] = hit_positions[:,0]
                data[2, hit_rows, hit_cols] = hit_positions[:,1]
                data[3, hit_rows, hit_cols] = hit_positions[:,2]

                # kill
                data[1, hit_rows_d, hit_cols_d] = .0
                data[2, hit_rows_d, hit_cols_d] = .0
                data[3, hit_rows_d, hit_cols_d] = .0
            
            if self.use_dead_pmt_mask:
                if self.use_positions:
                    data[4, hit_rows_d, hit_cols_d] = 1
                else:
                    data[1, hit_rows_d, hit_cols_d] = 1
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
            
            if self.use_dead_pmt_mask:
                if self.use_positions:
                    data[4, hit_rows_d, hit_cols_d] = 1
                else:
                    data[1, hit_rows_d, hit_cols_d] = 1

        return data
    


class CNNDatasetScale(CNNDatasetDeadPMT):
    """
    A dataset class for CNN models with additional scaling functionality.
    This class extends CNNDatasetDeadPMT to include features for scaling channel data,
    handling dead PMTs, and applying various scaling techniques to the time data.

    Attributes:
        dead_pmt_rate (float): Rate of dead PMTs to simulate.
        dead_pmt_seed (int): Seed for random number generation for dead PMTs.
        channel_scaler (dict): Configuration for scaling time channel.
        mask_train (numpy.ndarray): Boolean mask for training data.
        scaler: Scaler object or list of Scaler objects for time data transformation.
    """
    def __init__(self, h5file, pmt_positions_file, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None,
                 dead_pmt_rate=None, dead_pmt_seed=None, dead_pmts_file=None, use_dead_pmt_mask=False, channel_scaler=None, use_hit_mask=False):
        """
        Initialize the CNNDatasetScale object.
        """
        super().__init__(h5file, pmt_positions_file, use_times=use_times, use_charges=use_charges, use_positions=use_positions, transforms=transforms,
                         one_indexed=one_indexed, channel_scaling=channel_scaling, geometry_file=geometry_file,
                         dead_pmt_rate=dead_pmt_rate, dead_pmt_seed=dead_pmt_seed, dead_pmts_file=dead_pmts_file, use_dead_pmt_mask=use_dead_pmt_mask, use_hit_mask=use_hit_mask)

        self.channel_scaler = channel_scaler
        self.mask_train = None
        self.scale_time()
    
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
            print('------- get_train_mask (start) -------------')
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
            print('------- get_train_mask (end)-------------')
        
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

        self.scaler = []
        # apply scaler in order of list. So, to do normal(minmax(X)), list should look like ['minmax', 'normal']
        for idx, sclr in enumerate(self.channel_scaler['scaler_type']):
            self.scaler_status.append('')
            if 'minmax' in sclr:
                if sclr == 'minmax':
                    self.scaler.append(MinMaxScaler(feature_range=(0.1, 1.3)))
                else:
                    range_lo = re.findall(r'minmax_(\d+p?\d*).*', sclr)
                    range_hi = re.findall(r'minmax_.*_(\d+p?\d*)', sclr)
                    if len(range_lo) == 0 or len(range_hi) == 0:
                        print('Invalid minmax scaler type. Please provide range in the format: minmax_lo_hi')
                        print('lo and hi are numbers used for feature range. Example: minmax_0p1_1p3 for feature_range (0.1, 1.3)')
                        print('Fit minmax with (0.1, 1.3) instead')
                        self.scaler.append(MinMaxScaler(feature_range=(0.1, 1.3)))
                    else:
                        range_lo = float(range_lo[0].replace('p', '.'))
                        range_hi = float(range_hi[0].replace('p', '.'))
                        self.scaler.append(MinMaxScaler(feature_range=(range_lo, range_hi)))
            elif sclr == 'standard':
                self.scaler.append(StandardScaler())
            elif sclr == 'robust':
                self.scaler.append(RobustScaler())
            elif sclr == 'power':
                self.scaler.append(PowerTransformer())
            elif sclr == 'quantile_uniform':
                self.scaler.append(QuantileTransformer())
            elif sclr == 'quantile_normal':
                self.scaler.append(QuantileTransformer(output_distribution='normal'))
            else:
                print('Invalid scaler type. Please choose from: minmax, standard, robust, power, quantile_uniform, quantile_normal')
                return

            print('Will fit sk-learn scaler based on training set. Scaler:', sclr)

            mask_train = self.get_train_mask()

            print('Fitting the scaler on training set. This will take some time..')
            # if this scaler is second (or later) one to fit, transform the time data based on previous scaler(s)
            if idx > 0:
                print('-------------- Attention (fit_and_save_scaler)-------------------')
                print('Since you want to apply more than 1 scalers in sequence, we will transform the time data based on previous scaler first before fitting this one.')
                print('This will take some time...')
                # self.time = self.scaler[idx-1].transform(self.time.reshape(-1, 1)).reshape(-1,)

                print("before transformation: ", self.time[:10])

                if hasattr(self.scaler[idx], 'partial_fit'):
                    print('partial_fit available 1')
                    # for i in range(0, len(mask_train), 10000):
                    #     if i % 10000 == 0:
                    #         print('iteration', i)
                    #     time_batch = self.time[mask_train][i:i+10000].reshape(-1, 1)
                    #     for i in range(0, idx):
                    #         if self.scaler_status[i] == 'fitted':
                    #             time_batch = self.scaler[i].transform(time_batch).reshape(-1,)
                    #     self.scaler[idx].partial_fit(time_batch.reshape(-1, 1))
                else:
                    # has to be fitted on transformed copy of time data
                    for i in range(0, idx):
                        if self.scaler_status[i] == 'fitted':
                            self.time = self.scaler[i].transform(self.time.reshape(-1, 1)).reshape(-1,)
                            self.scaler_status[i] = 'transformed'

                print('Transformed the time data based on previous scaler; Returning to fit this scaler')
                print("after transformation: ", self.time[:10])
                print('-----------------------------------------------------------------')
            
            if hasattr(self.scaler[idx], 'partial_fit'):
                print('partial_fit available 2')
                step_size = 100000
                for i in range(0, np.sum(mask_train), step_size):
                    if i + step_size >= len(self.time[mask_train]):
                        # truncate
                        time_batch = self.time[mask_train][i:len(self.time[mask_train])].reshape(-1, 1)
                    else:
                        time_batch = self.time[mask_train][i:i+step_size].reshape(-1, 1)
                    if i % 2000000 == 0:
                        print('iteration', i)
                    for s in range(0, idx):
                        if self.scaler_status[s] == 'fitted':
                            time_batch = self.scaler[s].transform(time_batch).reshape(-1,)
                    self.scaler[idx].partial_fit(time_batch.reshape(-1, 1))
            else:
                self.scaler[idx].fit(self.time[mask_train].reshape(-1, 1))
            self.scaler_status[idx] = 'fitted'

            # saving the scaler object so that we can reuse it for another training and test (evaluation) dataset
            scaler_name_str = self.channel_scaler['scaler_type'][idx]
            
            # create directory if it does not exist
            if not os.path.exists(self.channel_scaler['scaler_output_path']):
                os.makedirs(self.channel_scaler['scaler_output_path'])
            joblib.dump(self.scaler[idx], f'{(self.channel_scaler["scaler_output_path"])}/{idx}_th_scaler_{scaler_name_str}.joblib')
            print(f'{idx} th scaler was fitted and saved')

            if debug_mode:
                print('Here is scaler fitted: ', self.scaler[idx].get_params())
        print('All scalers were fitted and saved')
    
    
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

        self.scaler_status = [] # either fitted or transformed

        # if scaler is provided, load the scaler, and will not re-fit the scaler(s)
        if self.channel_scaler['fitted_scaler'] is not None:
            # if type(self.channel_scaler['fitted_scaler']) == list:
            self.scaler = []
            for idx, scaler_joblib in enumerate(self.channel_scaler['fitted_scaler']):
                s = joblib.load(scaler_joblib)
                self.scaler.append(s)
                self.scaler_status.append('fitted')
                print(f'{idx} th scaler loaded', s.get_params())
        else:
            print('Fitting new scaler(s) based on training set')
            self.fit_and_save_scaler()
            print('Finished fitting scaler(s)')
        
        # apply scaling at once
        if not self.channel_scaler['transform_per_batch']:        
            print('Transforming the entire self.time (regardless of train/val/test split if such split exists) based on the scaler. This will take some time..')
            for idx, sclr in enumerate(self.scaler):
                if self.scaler_status[idx] == 'fitted':
                    scaler_name_str = self.channel_scaler['scaler_type'][idx]
                    self.time = self.scaler[idx].transform(self.time.reshape(-1, 1)).reshape(-1,)
                    self.scaler_status[idx] = 'transformed'
                    print(f'self.time was successfully scaled with {idx}th scaler:', self.channel_scaler['scaler_type'][idx])
        else:
            print('Transform data on per batch basis')

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
        
        """
        Linear scaler (X - offset) / scale is turned off.
        """
        # apply scaling to channels
        # for c, (offset, scale) in self.scaling.items():
        #     hit_data[c] = (hit_data[c] - offset)/scale


        # print('before scaling', hit_data['time'][:10], 'shape', hit_data['time'].shape)
        if self.scaler is not None:
            if self.channel_scaler['transform_per_batch']:
                if isinstance(self.scaler, list):
                    for ssc in self.scaler:
                        hit_data['time'] = ssc.transform(hit_data['time'].reshape(-1, 1)).reshape(-1,)
                else:
                    hit_data['time'] = self.scaler.transform(hit_data['time'].reshape(-1, 1)).reshape(-1,)
     
        if self.use_positions:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"], hit_positions=hit_data["position"]))
        else:
            processed_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data["time"], hit_data["charge"]))
        
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