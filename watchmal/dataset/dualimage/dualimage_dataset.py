"""
Here is a dataset class for loading dual-image data from HDF5 files.
"""

import h5py

# torch imports
from torch import from_numpy

# generic imports
import numpy as np

np.set_printoptions(threshold=np.inf)
# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.cnn.cnn_dataset import CNNDataset
import watchmal.dataset.data_utils as du

class DualImageDataset(CNNDataset):
    def __init__(
        self,
        h5file,
        pmt_positions_file,
        mpmt_positions_file=None,  
        num_valid_mpmt_modules=816,
        **kwargs,
    ):
        super().__init__(h5file, pmt_positions_file, **kwargs)

        self.num_valid_mpmt_modules = num_valid_mpmt_modules
        self.n_channels_mpmt = 38 

        self.mpmt_positions = None
        if mpmt_positions_file is not None:
            self.mpmt_positions = np.load(mpmt_positions_file)[
                "pmt_image_positions"
            ].astype(int)
        self.event_hits_index_mpmt = None
        self.hit_pmt_mpmt = None
        self.hit_time_mpmt = None
        self.hit_charge_mpmt = None

    def initialize(self):

        if self.initialized:
            return
        super().initialize()
        if "mpmt" not in self.h5_file:
            raise KeyError(f"'mpmt' group not found in HDF5 file {self.h5_path}")

        mpmt_group = self.h5_file["mpmt"]
        self.event_hits_index_mpmt = np.append(
            mpmt_group["event_hits_index"], mpmt_group["hit_pmt"].shape[0]
        ).astype(np.int64)

        if self.use_memmap:
            data = mpmt_group["hit_pmt"]
            self.hit_pmt_mpmt = np.memmap(
                self.h5_path,
                mode="r",
                shape=data.shape,
                offset=data.id.get_offset(),
                dtype=data.dtype,
            )

            data = mpmt_group["hit_time"]
            self.hit_time_mpmt = np.memmap(
                self.h5_path,
                mode="r",
                shape=data.shape,
                offset=data.id.get_offset(),
                dtype=data.dtype,
            )

            data = mpmt_group["hit_charge"]
            self.hit_charge_mpmt = np.memmap(
                self.h5_path,
                mode="r",
                shape=data.shape,
                offset=data.id.get_offset(),
                dtype=data.dtype,
            )
        else:
            self.hit_pmt_mpmt = np.array(mpmt_group["hit_pmt"])
            self.hit_time_mpmt = np.array(mpmt_group["hit_time"])
            self.hit_charge_mpmt = np.array(mpmt_group["hit_charge"])


    def _process_mpmt_data(self, hit_pmts, hit_times, hit_charges):
        sparse_data = np.zeros(
            (self.n_channels_mpmt, self.num_valid_mpmt_modules),
            dtype=np.float32,
        )

        if hit_pmts.size == 0:
            return sparse_data

        hit_pmts_int = hit_pmts.astype(int)
        if self.one_indexed:
            hit_pmts_int = hit_pmts_int - 1

        location_indices = hit_pmts_int // 19
        pmt_in_module_indices = hit_pmts_int % 19

        if self.use_log_charge:
            hit_charges = np.log10(hit_charges + 1e-6)

        time_channels = pmt_in_module_indices * 2
        charge_channels = pmt_in_module_indices * 2 + 1
        sparse_data[time_channels, location_indices] = hit_times
        sparse_data[charge_channels, location_indices] = hit_charges

        return sparse_data

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        data_main = data_dict.pop("data")  

        if not self.initialized:
            self.initialize()

        start_mpmt = self.event_hits_index_mpmt[item]
        stop_mpmt = self.event_hits_index_mpmt[item + 1]
        hit_pmts_mpmt = self.hit_pmt_mpmt[start_mpmt:stop_mpmt]
        hit_times_mpmt = self.hit_time_mpmt[start_mpmt:stop_mpmt]
        hit_charges_mpmt = self.hit_charge_mpmt[start_mpmt:stop_mpmt]

        sparse_mpmt_data_np = self._process_mpmt_data(
            hit_pmts_mpmt, hit_times_mpmt, hit_charges_mpmt
        )
        data_second = from_numpy(sparse_mpmt_data_np) 
        data_dict["data"] = (data_main, data_second)

        return data_dict