"""
Here is a dataset class for loading dual-image data from HDF5 files.
"""

# torch imports
from torch import from_numpy

# generic imports
import numpy as np

np.set_printoptions(threshold=np.inf)
# WatChMaL imports
from watchmal.dataset.cnn.cnn_dataset import CNNDataset

class DualImageDataset(CNNDataset):
    def __init__(
        self,
        h5file,
        pmt_positions_file,
        mpmt_positions_file=None,
        num_valid_mpmt_modules=816,
        pmt_type_main=0,
        pmt_type_mpmt=1,
        **kwargs,
    ):
        super().__init__(h5file, pmt_positions_file, pmt_type=pmt_type_main, **kwargs)

        self.num_valid_mpmt_modules = num_valid_mpmt_modules
        self.n_channels_mpmt = 38
        self.pmt_type_mpmt = pmt_type_mpmt

    def initialize(self):
        if self.initialized:
            return
        super().initialize()


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

        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]
        hit_pmts = self.hit_pmt[start:stop]
        hit_times = self.hit_time[start:stop]
        hit_charges = self.hit_charge[start:stop]
        hit_pmt_types = self.hit_pmt_type[start:stop] if self.hit_pmt_type is not None else None

        if self.pmt_type_mpmt is not None:
            if hit_pmt_types is None:
                print(f"WARNING: 'hit_pmt_type' not found in {self.h5_path}; mPMT hits set to empty")
                hit_pmts = hit_pmts[:0]
                hit_times = hit_times[:0]
                hit_charges = hit_charges[:0]
            else:
                if isinstance(self.pmt_type_mpmt, (list, tuple, set, np.ndarray)):
                    mask = np.isin(hit_pmt_types, self.pmt_type_mpmt)
                else:
                    mask = hit_pmt_types == self.pmt_type_mpmt
                hit_pmts = hit_pmts[mask]
                hit_times = hit_times[mask]
                hit_charges = hit_charges[mask]

        sparse_mpmt_data_np = self._process_mpmt_data(hit_pmts, hit_times, hit_charges)
        data_second = from_numpy(sparse_mpmt_data_np) 
        data_dict["data"] = (data_main, data_second)

        return data_dict
