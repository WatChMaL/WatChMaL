"""
Here is a dataset class for Double-Image T2T-ViT.
This class is adapted from CNNDataset to handle double images.
(Like the 20inch PMT images and mPMT images)
"""

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

    def __init__(self, h5file, pmt_positions_file, h5file_mpmt, mpmt_positions_file, **kwargs):
        super().__init__(h5file, pmt_positions_file, **kwargs)

        self.mpmt_dataset = H5Dataset(h5file_mpmt, kwargs.get("use_memmap", True))
        self.mpmt_positions = np.load(mpmt_positions_file)['pmt_image_positions'].astype(int)
        self.n_channels_mpmt = 38 
        data_size_mpmt_base = np.max(self.mpmt_positions, axis=0) + 1
        use_padding = kwargs.get("use_padding", False)
        if use_padding:
            padding_dim = kwargs.get("padding_to_fixed_dimension", [192, 192])
            data_size_mpmt_base = padding_dim
        self.data_size_mpmt = np.insert(data_size_mpmt_base, 0, self.n_channels_mpmt)

    def _process_mpmt_data(self, hit_pmts, hit_times, hit_charges):
        if self.one_indexed:
            hit_pmts = hit_pmts - 1

        hit_rows = self.mpmt_positions[hit_pmts, 0]
        hit_cols = self.mpmt_positions[hit_pmts, 1]

        time_offset = self.scale_offset.get("time", 0.0)
        time_scale = self.scale_factor.get("time", 1.0)
        charge_offset = self.scale_offset.get("charge", 0.0)
        charge_scale = self.scale_factor.get("charge", 1.0)
        
        if self.use_log_charge:
            hit_charges = np.log10(hit_charges)

        invalid_value = -100.0 if self.use_invalid_value else 0.0
        data = np.full(self.data_size_mpmt, invalid_value, dtype=np.float32)

        data[(hit_pmts % 19) * 2, hit_rows, hit_cols] = (hit_times - time_offset) / time_scale
        data[(hit_pmts % 19) * 2 + 1, hit_rows, hit_cols] = (hit_charges - charge_offset) / charge_scale

        return data

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        data_main = data_dict.pop('data')
        if not self.mpmt_dataset.initialized:
            self.mpmt_dataset.initialize()
            
        _ = self.mpmt_dataset[item]
        data_second = self._process_mpmt_data(
            self.mpmt_dataset.event_hit_pmts,
            self.mpmt_dataset.event_hit_times,
            self.mpmt_dataset.event_hit_charges
        )
        data_dict['data'] = (data_main, from_numpy(data_second))
        return data_dict
