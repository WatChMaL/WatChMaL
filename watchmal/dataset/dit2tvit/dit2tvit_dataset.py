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
import watchmal.dataset.data_utils as du


class DoubleImageDataset(H5Dataset):
    def __init__(
        self,
        h5file,
        h5file_mpmt,
        pmt_positions_file,
        mpmt_positions_file,
        use_times=True,
        use_charges=True,
        use_padding=False,
        padding_to_fixed_dimension=[192, 192],
        transforms=None,
        one_indexed=False,
        use_memmap=True,
        channel_scale_factor=None,
        channel_scale_offset=None,
        use_isHit=False,
        use_positions=False,
        use_orientations=False,
        geometry_file=None,
        use_invalid_value=False,
        use_log_charge=False,
    ):
        """
        Parameters:
        ----------
        h5file_mpmt: string
            Location of the HDF5 file containing the event data of the mPMT
        mpmt_positions_file: string
            Location of an npz file containing the mapping from mPMT IDs to CNN image pixel locations
        Other parameters are similar to those in CNNDataset.
        """
        super().__init__(h5file, use_memmap)
        self.mpmt_dataset = H5Dataset(h5file_mpmt, use_memmap)
        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(
            int
        )
        self.mpmt_positions = np.load(mpmt_positions_file)[
            "pmt_image_positions"
        ].astype(int)
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_isHit = use_isHit
        self.use_positions = use_positions
        self.use_orientations = use_orientations
        self.use_invalid_value = use_invalid_value
        self.use_log_charge = use_log_charge
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        self.data_size_mpmt = np.max(self.mpmt_positions, axis=0) + 1
        if use_padding:
            self.data_size = padding_to_fixed_dimension
            self.data_size_mpmt = padding_to_fixed_dimension

        self.barrel_rows = [
            row
            for row in range(self.data_size[0])
            if np.count_nonzero(self.pmt_positions[:, 0] == row) == self.data_size[1]
        ]
        self.transforms = None  # du.get_transformations(transformations, transforms)p
        self.one_indexed = one_indexed

        # TODO: geometry file for mPMT hasn't been down
        if use_positions:
            self.real_3Dpositions = np.load(geometry_file)["positions"]
        else:
            self.real_3Dpositions = None
        if use_orientations:
            self.real_3Dorientations = np.load(geometry_file)["orientations"]
        else:
            self.real_3Dorientations = None

        if channel_scale_offset is None:
            channel_scale_offset = {}
        self.scale_offset = channel_scale_offset
        if channel_scale_factor is None:
            channel_scale_factor = {}
        self.scale_factor = channel_scale_factor

        self.channel_map = {}
        current_channel = 0

        if use_times:
            self.channel_map["time"] = current_channel
            current_channel += 1

        if use_charges:
            self.channel_map["charge"] = current_channel
            current_channel += 1

        if use_isHit:
            self.channel_map["isHit"] = current_channel
            current_channel += 1

        if use_positions:
            self.channel_map["position_X"] = current_channel
            self.channel_map["position_Y"] = current_channel + 1
            self.channel_map["position_Z"] = current_channel + 2
            current_channel += 3

        if use_orientations:
            self.channel_map["orientation_X"] = current_channel
            self.channel_map["orientation_Y"] = current_channel + 1
            self.channel_map["orientation_Z"] = current_channel + 2
            current_channel += 3
        if "time" not in self.channel_map and "charge" not in self.channel_map:
            raise ValueError("No time or charge information loaded.")

        self.n_channels = current_channel
        self.n_channels_mpmt = 38
        self.data_size = np.insert(self.data_size, 0, self.n_channels)
        self.data_size_mpmt = np.insert(self.data_size_mpmt, 0, self.n_channels_mpmt)

    def process_data_main(self, hit_pmts, hit_times, hit_charges):
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
            hit_pmts = hit_pmts - 1  # SK cable numbers start at 1

        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]

        invalid_value = 0.0
        if self.use_invalid_value:
            invalid_value = -100.0

        time_offset = self.scale_offset.get("time", 0.0)
        time_scale = self.scale_factor.get("time", 1.0)
        charge_offset = self.scale_offset.get("charge", 0.0)
        charge_scale = self.scale_factor.get("charge", 1.0)
        positions_offset = self.scale_offset.get("positions", 0.0)
        positions_scale = self.scale_factor.get("positions", 1.0)
        orientations_offset = self.scale_offset.get("orientations", 0.0)
        orientations_scale = self.scale_factor.get("orientations", 1.0)

        if self.use_log_charge:
            hit_charges = np.log10(hit_charges)
        data = np.full(self.data_size, invalid_value, dtype=np.float32)
        if self.use_positions:
            data[
                self.channel_map["position_X"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 0] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Y"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 1] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Z"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 2] - positions_offset) / positions_scale
        if self.use_orientations:
            data[
                self.channel_map["orientation_X"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 0] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Y"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 1] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Z"],
                self.pmt_positions[:, 0],
                self.pmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 2] - orientations_offset
            ) / orientations_scale

        if "time" in self.channel_map:
            data[self.channel_map["time"], hit_rows, hit_cols] = (
                hit_times - time_offset
            ) / time_scale
        if "charge" in self.channel_map:
            data[self.channel_map["charge"], hit_rows, hit_cols] = (
                hit_charges - charge_offset
            ) / charge_scale

        if "isHit" in self.channel_map:
            data[self.channel_map["isHit"], hit_rows, hit_cols] = 1.0

        return data

    def process_data_second(self, hit_pmts, hit_times, hit_charges):
        """
        Returns second image (e.g., mPMT-based) event data formatted for CNN input.
        """
        if self.one_indexed:
            hit_pmts = hit_pmts - 1

        hit_rows = self.mpmt_positions[hit_pmts, 0]
        hit_cols = self.mpmt_positions[hit_pmts, 1]

        invalid_value = 0.0
        if self.use_invalid_value:
            invalid_value = -100.0

        time_offset = self.scale_offset.get("time", 0.0)
        time_scale = self.scale_factor.get("time", 1.0)
        charge_offset = self.scale_offset.get("charge", 0.0)
        charge_scale = self.scale_factor.get("charge", 1.0)
        positions_offset = self.scale_offset.get("positions", 0.0)
        positions_scale = self.scale_factor.get("positions", 1.0)
        orientations_offset = self.scale_offset.get("orientations", 0.0)
        orientations_scale = self.scale_factor.get("orientations", 1.0)

        if self.use_log_charge:
            hit_charges = np.log10(hit_charges)

        data = np.full(self.data_size_mpmt, invalid_value, dtype=np.float32)

        if self.use_positions:
            data[
                self.channel_map["position_X"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 0] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Y"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 1] - positions_offset) / positions_scale
            data[
                self.channel_map["position_Z"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (self.real_3Dpositions[:, 2] - positions_offset) / positions_scale

        if self.use_orientations:
            data[
                self.channel_map["orientation_X"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 0] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Y"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 1] - orientations_offset
            ) / orientations_scale
            data[
                self.channel_map["orientation_Z"],
                self.mpmt_positions[:, 0],
                self.mpmt_positions[:, 1],
            ] = (
                self.real_3Dorientations[:, 2] - orientations_offset
            ) / orientations_scale

        if "time" in self.channel_map:
            data[(hit_pmts % 19) * 2, hit_rows, hit_cols] = (
                hit_times - time_offset
            ) / time_scale
        if "charge" in self.channel_map:
            data[(hit_pmts % 19) * 2 + 1, hit_rows, hit_cols] = (
                hit_charges - charge_offset
            ) / charge_scale
        if "isHit" in self.channel_map:
            data[self.channel_map["isHit"], hit_rows, hit_cols] = 1.0

        return data

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()
        if not self.mpmt_dataset.initialized:
            self.mpmt_dataset.initialize()
        data_dict = super().__getitem__(item)

        data_main = from_numpy(
            self.process_data_main(
                self.event_hit_pmts, self.event_hit_times, self.event_hit_charges
            )
        )
        if self.transforms:
            data_main = du.apply_random_transformations(self.transforms, data_main)
        mpmt_record = self.mpmt_dataset[item]
        data_second = from_numpy(
            self.process_data_second(
                self.mpmt_dataset.event_hit_pmts,
                self.mpmt_dataset.event_hit_times,
                self.mpmt_dataset.event_hit_charges,
            )
        )

        data_dict["data_main"] = data_main  # e.g. shape [2, 192, 192]
        data_dict["data_second"] = data_second  # e.g. shape [38, 192, 192]

        return data_dict
