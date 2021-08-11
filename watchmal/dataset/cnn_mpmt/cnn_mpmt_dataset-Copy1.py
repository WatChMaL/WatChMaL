"""
Class implementing a mPMT dataset for CNNs in h5 format
"""

# torch imports
from torch import from_numpy
from torch import flip

# generic imports
import numpy as np
import torch

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.cnn_mpmt import transformations
import watchmal.dataset.data_utils as du

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19

class CNNmPMTDataset(H5Dataset):
    def __init__(self, h5file, mpmt_positions_file, is_distributed, transforms=None, collapse_arrays=False, pad=False):
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
#         self.transforms = du.get_transformations(transformations, transforms)
        self.pad = pad
        
        ################
        self.transforms = transforms
        self.transforms_funcs = {
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': , self.vertical_flip,
            'rotation180': self.rotation180
        }
        self.horizontal_flip_mpmt_map=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map=[6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]
        ################

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

        data = np.zeros(self.data_size)
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
        
        processed_data = self.apply_random_transformations(processed_data)
        
        # Add padding
        if self.pad:
            processed_data = transformations.mpmtPadding(processed_data, self.barrel_rows)
            
        data_dict["data"] = processed_data

        return data_dict
    
    def apply_random_transformations(self, data, segmented_labels = None):
        if self.transforms is not None:
            for transformation in self.transforms:
                if random.getrandbits(1):
                    data = self.transforms_funcs[transformation](data)
                    if segmented_labels is not None:
                        segmented_labels = self.transforms_funcs[transformation](segmented_labels)
        return data
    
    def horizontal_flip(self, data):
        return flip(data[horizontal_flip_mpmt_map, :, :, ], [2])

    def vertical_flip(self, data):
        return flip(data[vertical_flip_mpmt_map, :, :], [1])
    
    def rotation180(self, data, barrel_rows):
        transform_data = data.clone()
        transform_data[:, barrel_rows, :] = torch.tensor(np.roll(transform_data[:, barrel_rows, :], 20, 2))

        l_index = data.shape[2]/2 - 1
        r_index = data.shape[2]/2

        l_endcap_ind = int(l_index - 4)
        r_endcap_ind = int(r_index + 4)

        top_end_cap = data.clone()[:, barrel_rows[-1]+1:, l_endcap_ind:r_endcap_ind+1]
        bot_end_cap = data.clone()[:, :barrel_rows[0], l_endcap_ind:r_endcap_ind+1]

        vhflip_top = horizontal_flip(vertical_flip(top_end_cap))
        vhlfip_bot = horizontal_flip(vertical_flip(bot_end_cap))

        transform_data[:, barrel_rows[-1]+1:, l_endcap_ind : r_endcap_ind+1] = vhflip_top
        transform_data[:, :barrel_rows[0], l_endcap_ind : r_endcap_ind+1] = vhlfip_bot

        return transform_data
    
    def mpmtPadding(self, data, barrel_rows):
        half_len_index = int(data.shape[2]/2)
        horiz_pad_data = torch.cat((data, torch.zeros_like(data[:, :, :half_len_index])), 2)
        horiz_pad_data[:, :, 2*half_len_index:] = torch.tensor(0, dtype=torch.float64)
        horiz_pad_data[:, barrel_rows, 2*half_len_index:] = data[:, barrel_rows, :half_len_index]

        l_index = data.shape[2]/2 - 1
        r_index = data.shape[2]/2

        l_endcap_ind = int(l_index - 4)
        r_endcap_ind = int(r_index + 4)

        top_end_cap = data[:, barrel_rows[-1]+1:, l_endcap_ind:r_endcap_ind+1]
        bot_end_cap = data[:, :barrel_rows[0], l_endcap_ind:r_endcap_ind+1]

        vhflip_top = horizontal_flip(vertical_flip(top_end_cap))
        vhflip_bot = horizontal_flip(vertical_flip(bot_end_cap))

        horiz_pad_data[:, barrel_rows[-1]+1:, l_endcap_ind + int(data.shape[2]/2) : r_endcap_ind + int(data.shape[2]/2) + 1] = vhflip_top
        horiz_pad_data[:, :barrel_rows[0], l_endcap_ind + int(data.shape[2]/2) : r_endcap_ind + int(data.shape[2]/2) + 1] = vhflip_bot

        return horiz_pad_data
    
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
