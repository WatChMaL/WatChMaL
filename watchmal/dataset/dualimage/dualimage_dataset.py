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
        mpmt_positions_file,
        num_valid_mpmt_modules=816,
        **kwargs,
    ):
        super().__init__(h5file, pmt_positions_file, **kwargs)

        self.num_valid_mpmt_modules = num_valid_mpmt_modules
        self.mpmt_positions = np.load(mpmt_positions_file)[
            "pmt_image_positions"
        ].astype(int)

        self.n_channels_mpmt = 38
        data_size_mpmt_base = np.max(self.mpmt_positions, axis=0) + 1

        use_padding = kwargs.get("use_padding", False)
        if use_padding:
            padding_dim = kwargs.get("padding_to_fixed_dimension", [192, 192])
            data_size_mpmt_base = padding_dim

        self.data_size_mpmt = np.insert(data_size_mpmt_base, 0, self.n_channels_mpmt)

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

        tube_idx = hit_pmts % 19
        time_channels = tube_idx * 2
        charge_channels = tube_idx * 2 + 1

        data[time_channels, hit_rows, hit_cols] = (hit_times - time_offset) / time_scale
        data[charge_channels, hit_rows, hit_cols] = (
            hit_charges - charge_offset
        ) / charge_scale

        return data

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        data_main = data_dict.pop("data")

        start_mpmt = self.event_hits_index_mpmt[item]
        stop_mpmt = self.event_hits_index_mpmt[item + 1]

        hit_pmts_mpmt = self.hit_pmt_mpmt[start_mpmt:stop_mpmt]
        hit_times_mpmt = self.hit_time_mpmt[start_mpmt:stop_mpmt]
        hit_charges_mpmt = self.hit_charge_mpmt[start_mpmt:stop_mpmt]

        data_second_np = self._process_mpmt_data(
            hit_pmts_mpmt, hit_times_mpmt, hit_charges_mpmt
        )
        data_second = from_numpy(data_second_np)

        data_dict["data"] = (data_main, data_second)
        return data_dict
