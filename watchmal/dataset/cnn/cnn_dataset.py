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
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

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
        if False:
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

