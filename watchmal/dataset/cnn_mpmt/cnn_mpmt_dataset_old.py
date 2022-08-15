"""
Class implementing a mPMT dataset for CNNs in h5 format
"""

# torch imports
from torch import from_numpy
from torch import flip

# generic imports
import numpy as np
import torch
import random

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19

class CNNmPMTDataset(H5Dataset):
    def __init__(self, h5file, mpmt_positions_file, is_distributed, padding_type=None, transforms=None, collapse_arrays=False):
        """
        Args:
            h5_path             ... path to h5 dataset file
            is_distributed      ... whether running in multiprocessing mode
            transforms          ... transforms to apply
            collapse_arrays     ... whether to collapse arrays in return
        """
        super().__init__(h5file, is_distributed)
        
        
        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.data_size = np.max(self.mpmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.mpmt_positions[:,0] == row) == self.data_size[1]]
        n_channels = pmts_per_mpmt
        self.data_size = np.insert(self.data_size, 0, n_channels)
        self.collapse_arrays = collapse_arrays
        self.transforms = du.get_transformations(self, transforms)

        if padding_type is not None:
            self.padding_type = getattr(self, padding_type)
        else:
            self.padding_type = lambda x : x 

        self.horizontal_flip_mpmt_map=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map=[6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]


    def process_data(self, hit_pmts, hit_data):
        """
        Returns event data from dataset associated with a specific index
        Args:
            hit_pmts                ... array of ids of hit pmts
            hid_data                ... array of data associated with hits
        
        Returns:
            data                    ... array of hits in cnn format
        """
        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

        # fix barrel array indexing to match endcaps in xyz ordering
        barrel_data = data[:, self.barrel_rows, :]
        data[:, self.barrel_rows, :] = barrel_data[barrel_map_array_idxs, :, :]

        # collapse arrays if desired
        if self.collapse_arrays:
            data = np.expand_dims(np.sum(data, 0), 0)
        
        return data

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)

        processed_data = from_numpy(self.process_data(self.event_hit_pmts, self.event_hit_charges))
        
        processed_data = du.apply_random_transformations(self.transforms, processed_data)

        processed_data = self.padding_type(processed_data)
            
        data_dict["data"] = processed_data
        
        return data_dict
        

    def horizontal_flip(self, data):
        return flip(data[self.horizontal_flip_mpmt_map, :, :], [2])

    def vertical_flip(self, data):
        return flip(data[self.vertical_flip_mpmt_map, :, :], [1])

    def flip_180(self, data):
        return self.horizontal_flip(self.vertical_flip(data))
 
    def front_back_reflection(self, data):
        """
        Returns an image with horizontal flip of the left and right halves of the barrels and
        vertical flip of the endcaps
        Specs in transform.pdf

        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """

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
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data


    def rotation180(self, data):
        """
        Returns an image with horizontal and vertical flip of the endcaps and 
        shifting of the barrel rows by half the width
        Specs in transforms.pdf

        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25   

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:,self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data
    
    
    def mpmtPadding(self, data):
        """
        :param data: torch.tensor
        :returns transform_data: torch.tensor
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index+1]

        padded_data[:, :barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data


    def double_cover(self, data):
        """
        Specs in double_cover.pdf

        param data: torch.tensor
        returns padded_data: torch.tensor
        """
        w = data.shape[2]                                                                            
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                                                    # 5
        half_barrel_width, quarter_barrel_width = w//2, w//4                                   # 20, 10

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180 
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width               #  5
        r1_endcap_index = l1_endcap_index + 2*radius_endcap                                       # 15
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1: , l1_endcap_index:r1_endcap_index]
        
        padded_data[:, :barrel_row_start , l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)
        
        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]
        
        concat_order = (self.flip_180(barrel_top_half), 
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data


    def retrieve_event_data(self, item):
        """
        Returns event data from dataset associated with a specific index
        Args:
            item                    ... index of event
        Returns:
            hit_pmts                ... array of ids of hit pmts
            pmt_charge_data         ... array of charge of hits
            pmt_time_data           ... array of times of hits
        """
        data_dict = super().__getitem__(item)

        # construct charge data with barrel array indexing to match endcaps in xyz ordering
        pmt_charge_data = self.process_data(self.event_hit_pmts, self.event_hit_charges).flatten()

        # construct time data with barrel array indexing to match endcaps in xyz ordering
        pmt_time_data = self.process_data(self.event_hit_pmts, self.event_hit_times).flatten()

        return self.event_hit_pmts, pmt_charge_data, pmt_time_data
        
