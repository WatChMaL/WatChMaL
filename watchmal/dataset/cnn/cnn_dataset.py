"""
Class implementing a PMT dataset for CNNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
from torch import from_numpy

# generic imports
import numpy as np

np.set_printoptions(threshold=np.inf)
# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du


class CNNDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit time and/or charge, and the second and third dimensions
    are the height and width of the CNN image. Each pixel of the image corresponds to one PMT, with PMTs arrange in an
    event-display-like format.
    """

    def __init__(
        self,
        h5file,
        pmt_positions_file,
        use_times=True,
        use_charges=True,
        use_padding=False,
        padding_to_fixed_dimension=[192, 192],
        transforms=None,
        one_indexed=False,
        use_memmap=True,
        mask_pmts=None,
        channel_scale_factor=None,
        channel_scale_offset=None,
        use_isHit=False,
        use_positions=False,
        use_orientations=False,
        geometry_file=None,
        use_invalid_value=False,
        use_median_unhit_times=False,
        use_log_charge=False,
    ):
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
        use_memmap: bool
            Use a memmap and load data into memory as needed (default), otherwise load entire dataset at initialisation
        mask_pmts: list of int
            List of PMT IDs to mask out from all data (None by default)
        ---------- The following features are used for optimization.
        channel_scale_factor: dict of float
            Dictionary with keys corresponding to channels and values contain the factors to divide that channel.
            By default, no scaling is applied.
        channel_scale_offset: dict of float
            Dictionary with keys corresponding to channels and values contain the offsets to subtract from that channel.
            By default, no scaling is applied.
        use_isHit: bool
            Whether to use a channel to tag the PMT hit or not.
        use_positions: bool
            Whether to use three channels to add the real positions info of PMTs.
        use_orientations: bool
            Whether to use three channels to add the real orientations info of PMTs.
        geometry_file: string
            Location of an npz file containing the real positions and orientations info.
        use_invalid_value: bool
            Whether to set all the channel of unhit as an invalid value (like -100).
        use_median_unhit_times: bool
            Whether to set unhit times to the median value of the normalised hit times
        use_log_charge: bool
            Whether to logarithmically transform the charge.
        ---------- The following features are used for Visual Transformer. 
        Because in ViT we need to devide the image, the original image size is 191*191 (HK 20inch PMT), so here we directly pad the image to 192*192, which is more easy to be divided. 
        use_padding: bool
            Whether to pad the data to a fixed dimension (default: False).
        padding_to_fixed_dimension: list of int
            If use_padding is True, this specifies the fixed dimension to which the data will be padded.
            Default: [192, 192] (for 192x192 images).
        ----------
        """

        super().__init__(h5file, use_memmap, mask_pmts)

        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(
            int
        )
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_isHit = use_isHit
        self.use_positions = use_positions
        self.use_orientations = use_orientations
        self.use_invalid_value = use_invalid_value
        self.use_median_unhit_times = use_median_unhit_times
        self.use_log_charge = use_log_charge
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        if use_padding:
            self.data_size = padding_to_fixed_dimension

        self.image_height, self.image_width = self.data_size[0], self.data_size[1]
        # make some index expressions for different parts of the image, to use in transformations etc
        rows, row_counts = np.unique(self.pmt_positions[:, 0], return_counts=True)
        cols, col_counts = np.unique(self.pmt_positions[:, 1], return_counts=True)
        # barrel rows are those where the row appears in mpmt_positions as many times as the image width
        barrel_rows = rows[row_counts > 0.7 * self.image_width]
        # endcap_size is the number of rows before the first barrel row
        self.endcap_size = np.min(barrel_rows)
        self.barrel = np.s_[..., self.endcap_size:np.max(barrel_rows) + 1, :]
        # endcap columns are those where the column appears more than the number of barrel rows
        endcap_cols = cols[col_counts > len(barrel_rows)]
        self.endcap_left = np.min(endcap_cols)
        self.endcap_right = np.max(endcap_cols) + 1
        self.top_endcap = np.s_[..., :self.endcap_size, self.endcap_left:self.endcap_right]
        self.bottom_endcap = np.s_[..., -self.endcap_size:, self.endcap_left:self.endcap_right]


        self.transforms = du.get_transformations(self, transforms)
        if self.transforms is None:
            self.transforms = []

        self.one_indexed = one_indexed

        if use_positions:
            self.real_3Dpositions = np.load(geometry_file)["position"]
        else:
            self.real_3Dpositions = None
        if use_orientations:
            self.real_3Dorientations = np.load(geometry_file)["orientation"]
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
        self.data_size = np.insert(self.data_size, 0, self.n_channels)

    def process_data(self, hit_pmts, hit_times, hit_charges):
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
            normalised_times = (hit_times - time_offset) / time_scale
            if self.use_median_unhit_times:
                median_time = np.median(normalised_times)
                data[self.channel_map["time"]] = median_time
            data[self.channel_map["time"], hit_rows, hit_cols] = normalised_times
        if "charge" in self.channel_map:
            data[self.channel_map["charge"], hit_rows, hit_cols] = (
                hit_charges - charge_offset
            ) / charge_scale
        if "isHit" in self.channel_map:
            data[self.channel_map["isHit"], hit_rows, hit_cols] = 1.0

        return data

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        processed_data = self.process_data(self.event_hit_pmts, self.event_hit_times, self.event_hit_charges)

        # Apply transformations
        data_dict["data"] = processed_data
        for t in self.transforms:
            data_dict = t(data_dict)
        data_dict["data"] = from_numpy(data_dict["data"].copy())

        return data_dict

    def double_cover(self, data_dict):
        """
        Takes CNN input data in event-display-like format and returns the data with all parts of the detector duplicated
        and rearranged to provide a double-cover of the image, providing two 'views' of the detector from a single image
        with less blank space and physically meaningful cyclic boundary conditions at the edges of the image.

        Since CNNDataset uses a simple PMT grid (1 PMT per pixel) instead of mPMT (19 PMTs per pixel), this version
        is simpler - no channel permutations are needed.

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
        # Make copies of the endcaps, flipped (180° rotated), to use later
        top_endcap_copy = np.flip(data_dict["data"][self.top_endcap], [1, 2])
        bottom_endcap_copy = np.flip(data_dict["data"][self.bottom_endcap], [1, 2])
        # Roll the tensor so that the first quarter is the last quarter
        quarter_barrel_width = self.image_width // 4
        data = np.roll(data_dict["data"], -quarter_barrel_width, 2)
        # Paste the copied flipped endcaps a quarter barrel-width past the original endcap position
        endcap_copy_columns = np.s_[quarter_barrel_width + self.endcap_left: quarter_barrel_width + self.endcap_right]
        data[..., :self.endcap_size, endcap_copy_columns] = top_endcap_copy
        data[..., -self.endcap_size:, endcap_copy_columns] = bottom_endcap_copy
        # Rotate the bottom and top halves of barrel and concatenate to the top and bottom of the image
        # If the endcaps are offset from the middle of the image, need to roll the flipped barrel to keep the same offset
        offset = (self.image_width - self.endcap_right) - self.endcap_left
        barrel_rolled = np.roll(data[self.barrel], offset, 2)
        barrel_bottom_flipped, barrel_top_flipped = np.array_split(np.flip(barrel_rolled, [1, 2]), 2, axis=1)
        data_dict["data"] = np.concatenate((barrel_top_flipped, data, barrel_bottom_flipped), axis=1)
        return data_dict

