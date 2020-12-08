# PyTorch imports
from torch.utils.data import Dataset
import h5py
import numpy as np
from abc import ABC, abstractmethod
import copy
import time

import torch.multiprocessing as mp

class H5Dataset(Dataset, ABC):

    def __init__(self, h5_path, is_distributed, transforms=None):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_length = h5_file["labels"].shape[0]

            hdf5_hit_pmt    = h5_file["hit_pmt"]
            hdf5_hit_time   = h5_file["hit_time"]
            hdf5_hit_charge = h5_file["hit_charge"]

            # initialize memmap param dict
            self.pmt_dict    = {'shape':hdf5_hit_pmt.shape,    'offset':hdf5_hit_pmt.id.get_offset(),   'dtype':hdf5_hit_pmt.dtype}
            self.time_dict   = {'shape':hdf5_hit_time.shape,   'offset':hdf5_hit_time.id.get_offset(),  'dtype':hdf5_hit_time.dtype}
            self.charge_dict = {'shape':hdf5_hit_charge.shape, 'offset':hdf5_hit_charge.id.get_offset(),'dtype':hdf5_hit_charge.dtype}
        
        self.initialized = False
        if not is_distributed:
            self.initialize()

    def initialize(self):
        """
        memmaps must be instantiated this way for multiprocessing (memmaps can't be pickled)
        """
        # Create a memory map for event_data - loads event data into memory only on __getitem__()
        self.hit_pmt = np.memmap(self.h5_path, mode="r",
                                shape=self.pmt_dict['shape'],
                                offset=self.pmt_dict['offset'],
                                dtype=self.pmt_dict['dtype'])
        
        self.time = np.memmap(self.h5_path, mode="r",
                                shape=self.time_dict['shape'],
                                offset=self.time_dict['offset'],
                                dtype=self.time_dict['dtype'])

        self.charge = np.memmap(self.h5_path, mode="r",
                                shape=self.charge_dict['shape'],
                                offset=self.charge_dict['offset'],
                                dtype=self.charge_dict['dtype'])
        
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.labels           = np.array(h5_file["labels"])
            self.energies         = np.array(h5_file["energies"])
            self.positions        = np.array(h5_file["positions"])
            self.angles           = np.array(h5_file["angles"])
            self.root_files       = np.array(h5_file["root_files"])
            self.event_ids        = np.array(h5_file["event_ids"])
            self.event_hits_index = np.append(h5_file["event_hits_index"], self.pmt_dict['shape'][0]).astype(np.int64)
        
        # Set attribute so that method won't be invoked again
        self.initialized = True

    @abstractmethod
    def get_data(self, hit_pmts, hit_charges, hit_times):
        pass

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()
        
        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        hit_pmts = self.hit_pmt[start:stop].astype(np.int16)
        hit_charges = self.charge[start:stop]
        hit_times = self.time[start:stop]

        data = self.get_data(hit_pmts, hit_charges, hit_times)
        
        data_dict = {
            "data": data,
            "labels": self.labels[item],
            "energies": self.energies[item],
            "angles": self.angles[item],
            "positions": self.positions[item],
            "root_files": self.root_files[item],
            "event_ids": self.event_ids[item],
            "indices": item
        }

        return data_dict
