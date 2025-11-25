"""
Class implementing a mPMT dataset for CNNs in h5 format
"""

# generic imports
import numpy as np
import torch

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

PMTS_PER_MPMT = 19
# maps to permute the PMTs within mPMT for various transformations, etc.
# Old convention starts with outer ring and works inwards, with mPMT oriented with a vertical line of PMTs in the mPMT
#           06
#      07        05
#  08       15      04
#      16        14
# 09        18        03
#      17        13
#   10      12      02
#      11        01
#           00
BARREL_MPMT_MAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18])
VERTICAL_FLIP_MPMT_MAP = np.array([6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18])
HORIZONTAL_FLIP_MPMT_MAP = np.array([0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18])
# New convention starts with central PMT and works outwards, with mPMT oriented with a horizontal line of PMTs in the mPMT
#           10
#       11      09
#   12    03  02    08

# 13   04   00   01   07
#
#   14    05  06    18
#       15      17
#           16
BARREL_MPMT_MAP_NEW = np.array([0, 4, 5, 6, 1, 2, 3, 13, 14, 15, 16, 17, 18, 7, 8, 9, 10, 11, 12])
VERTICAL_FLIP_MPMT_MAP_NEW = np.array([0, 1, 6, 5, 4, 3, 2, 7, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8])
HORIZONTAL_FLIP_MPMT_MAP_NEW = np.array([0, 4, 3, 2, 1, 6, 5, 13, 12, 11, 10, 9, 8, 7, 18, 17, 16, 15, 14])


class CNNmPMTDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit charge of the 19 PMTs within each mPMT, and the second
    and third dimensions are the height and width of the CNN image. Each pixel of the image corresponds to one mPMT,
    with mPMTs arrange in an event-display-like format.
    """

    def __init__(self, h5file, mpmt_positions_file, transforms=None, use_new_mpmt_convention=False, channels=None,
                 collapse_mpmt_channels=None, channel_scale_factor=None, channel_scale_offset=None, geometry_file=None,
                 use_memmap=True):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge data is
        formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to one mPMT
        module, with channels corresponding to each PMT within the mPMT. The mPMTs are placed in the image according to
        a mapping provided by the numpy array in the `mpmt_positions_file`.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        mpmt_positions_file: string
            Location of a npz file containing the mapping from mPMT IDs to CNN image pixel locations
        transforms: sequence of string
            List of random transforms to apply to data before passing to CNN for data augmentation. Each element of the
            list should be the name of a method of this class that performs the transformation
        use_new_mpmt_convention: bool
            Whether to use the new or old (default) convention of WCSim for how PMT channels are mapped within the mPMT.
        channels: sequence of string
            List defines the PMT data included in the image-like CNN arrays. It can be either 'charge', 'time' or both
            (default)
        collapse_mpmt_channels: sequence of string
            List of the data to be collapsed to two channels, containing, respectively, the mean and the std of other
            channels. i.e. provides the mean and the std of PMT charges and/or time in each mPMT instead of providing
            all PMT data. It can be [], ['charge'], ['time'] or ['charge', 'time']. By default, no collapsing is
            performed.
        channel_scale_factor: dict of float
            Dictionary with keys corresponding to channels and values contain the factors to divide that channel by.
            By default, no scaling is applied.
        channel_scale_offset: dict of float
            Dictionary with keys corresponding to channels and values contain the offsets to subtract from that channel.
            By default, no scaling is applied.
        geometry_file: string
            Path to file defining the 3D geometry (PMT positions and orientations), required if using geometry channels.
        use_memmap: bool
            Use a memmap and load data into memory as needed (default), otherwise load entire dataset at initialisation
"""

        super().__init__(h5file, use_memmap)

        self.use_new_mpmt_convention = use_new_mpmt_convention
        if self.use_new_mpmt_convention:
            self.barrel_mpmt_map = BARREL_MPMT_MAP_NEW
            self.vertical_flip_mpmt_map = VERTICAL_FLIP_MPMT_MAP_NEW
            self.horizontal_flip_mpmt_map = HORIZONTAL_FLIP_MPMT_MAP_NEW
        else:
            self.barrel_mpmt_map = BARREL_MPMT_MAP
            self.vertical_flip_mpmt_map = VERTICAL_FLIP_MPMT_MAP
            self.horizontal_flip_mpmt_map = HORIZONTAL_FLIP_MPMT_MAP
        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.transforms = du.get_transformations(self, transforms)
        if self.transforms is None:
            self.transforms = []
        if channels is None:
            channels = ['charge', 'time']
        if collapse_mpmt_channels is None:
            collapse_mpmt_channels = []
        self.collapse_channels = collapse_mpmt_channels
        if channel_scale_offset is None:
            channel_scale_offset = {}
        self.scale_offset = channel_scale_offset
        if channel_scale_factor is None:
            channel_scale_factor = {}
        self.scale_factor = channel_scale_factor

        self.image_height, self.image_width = np.max(self.mpmt_positions, axis=0) + 1
        # make some index expressions for different parts of the image, to use in transformations etc
        rows, row_counts = np.unique(self.mpmt_positions[:, 0], return_counts=True)  # count occurrences of each row
        cols, col_counts = np.unique(self.mpmt_positions[:, 1], return_counts=True)  # count occurrences of each column
        # barrel rows are those where the row appears in mpmt_positions as many times as the image width
        barrel_rows = rows[row_counts == self.image_width]
        # endcap size is the number of rows before the first barrel row
        self.endcap_size = np.min(barrel_rows)
        self.barrel = np.s_[..., self.endcap_size:np.max(barrel_rows) + 1, :]
        # endcaps are assumed to be within squares above and below the barrel
        # endcap columns are those where the column appears in mpmt_positions more than the number of barrel rows
        self.endcap_left = np.min(cols[col_counts > len(barrel_rows)])
        self.endcap_right = self.endcap_left + self.endcap_size
        self.top_endcap = np.s_[..., :self.endcap_size, self.endcap_left:self.endcap_right]
        self.bottom_endcap = np.s_[..., -self.endcap_size:, self.endcap_left:self.endcap_right]

        # encode the 3D geometry into optional extra CNN input channels
        self.geom_data = {}
        if geometry_file is None:
            if not set(channels).isdisjoint({"mpmt_position", "mpmt_direction", "mpmt_exists"}):
                raise TypeError("A geometry file must be provided if using channels that encode the geometry.")
        else:
            geo_file = np.load(geometry_file)
            pmt_positions = geo_file['position']
            pmt_directions = geo_file['orientation']
            pmt_ids = np.arange(0, pmt_positions.shape[0])
            central_pmt_channel = 0 if self.use_new_mpmt_convention else 18
            self.geom_data['mpmt_position'] = self.process_data(pmt_ids, pmt_positions)[central_pmt_channel]
            self.geom_data['mpmt_direction'] = self.process_data(pmt_ids, pmt_directions)[central_pmt_channel]
            self.geom_data['mpmt_exists'] = self.process_data(pmt_ids, 1)[central_pmt_channel]
            for c, v in self.geom_data.items():
                self.geom_data[c] = (v - self.scale_offset.pop(c, 0))/self.scale_factor.pop(c, 1)

        # set up data ranges and permutation maps for the chosen channels
        self.image_depth = 0
        # dictionary of ranges for to the slices of the CNN tensor that correspond to each given channel
        self.channel_ranges = {}
        # permutation of channels (ie PMTs within mPMT) when flipping the image horizontally
        self.h_flip_permutation = []
        # permutation of channels (ie PMTs within mPMT) when flipping the image vertically
        self.v_flip_permutation = []
        # which channels are actual data channels (not fixed geometry encoding) to get transformed when flipping image
        self.data_channels = []
        for c in channels:
            channel_depth = (self.geom_data[c].shape[0] if c in self.geom_data
                             else 2 if c in collapse_mpmt_channels
                             else PMTS_PER_MPMT)
            self.channel_ranges[c] = range(self.image_depth, self.image_depth+channel_depth)
            if c not in self.geom_data:
                self.data_channels.extend(self.channel_ranges[c])
            # permutation maps are needed for applying transformations to the image that affect mPMT channel ordering
            if channel_depth == PMTS_PER_MPMT:
                self.v_flip_permutation.extend(self.vertical_flip_mpmt_map + self.image_depth)
                self.h_flip_permutation.extend(self.horizontal_flip_mpmt_map + self.image_depth)
            else:
                self.v_flip_permutation.extend(self.channel_ranges[c])
                self.h_flip_permutation.extend(self.channel_ranges[c])
            self.image_depth += channel_depth
        self.data_channels = np.array(self.data_channels)
        self.h_flip_permutation = np.array(self.h_flip_permutation)
        self.v_flip_permutation = np.array(self.v_flip_permutation)
        self.rotate_permutation = self.h_flip_permutation[self.v_flip_permutation]

    def process_data(self, pmts, pmt_data):
        """Returns image-like event data array (channels, rows, columns) from arrays of PMT IDs and data at the PMTs."""
        # Ensure the data is a 2D array with first dimension being the number of pmts
        pmt_data = np.atleast_1d(pmt_data)
        pmt_data = pmt_data.reshape(pmt_data.shape[0], -1)

        mpmts = pmts // PMTS_PER_MPMT
        channels = pmts % PMTS_PER_MPMT

        rows = self.mpmt_positions[mpmts, 0]
        cols = self.mpmt_positions[mpmts, 1]

        data = np.zeros((PMTS_PER_MPMT, pmt_data.shape[1], self.image_height, self.image_width), dtype=np.float32)
        data[channels, :, rows, cols] = pmt_data

        # fix indexing of barrel PMTs in mPMT modules to match that of endcaps in the projection to 2D
        data[self.barrel] = data[self.barrel_mpmt_map][self.barrel]

        return data

    def __getitem__(self, item):
        """Returns image-like event data array (channels, rows, columns) for an event at a given index."""
        data_dict = super().__getitem__(item)
        hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times}
        # apply scaling to channels
        for c in hit_data:
            hit_data[c] = (hit_data[c] - self.scale_offset.get(c, 0))/self.scale_factor.get(c, 1)
        # Process the channels
        data = np.zeros((self.image_depth, self.image_height, self.image_width), dtype=np.float32)
        for c, r in self.channel_ranges.items():
            if c in hit_data:
                channel_data = self.process_data(self.event_hit_pmts, hit_data[c]).squeeze()
                if c in self.collapse_channels:
                    channel_data = collapse_channel(channel_data)
            elif c in self.geom_data:
                channel_data = self.geom_data[c]
            else:
                raise ValueError(f"Channel '{c}' is not available.")
            data[r] = channel_data
        # Apply transformations
        data_dict["data"] = data
        for t in self.transforms:
            data_dict = t(data_dict)
        data_dict["data"] = torch.from_numpy(data_dict["data"].copy())
        return data_dict

    def horizontal_image_flip(self, data):
        """Perform horizontal flip of image data, permuting mPMT channels where needed."""
        return np.flip(data[self.h_flip_permutation, :, :], [2])

    def vertical_image_flip(self, data):
        """Perform vertical flip of image data, permuting mPMT channels where needed."""
        return np.flip(data[self.v_flip_permutation, :, :], [1])

    def rotate_image(self, data):
        """Perform 180 degree rotation of image data, permuting mPMT channels where needed."""
        return np.flip(data[self.rotate_permutation, :, :], [1, 2])

    def horizontal_reflection(self, data_dict):
        """Takes CNN input data and truth info and performs horizontal flip, permuting mPMT channels where needed."""
        data_dict["data"][self.data_channels] = self.horizontal_image_flip(data_dict["data"])[self.data_channels]
        # If the endcaps are offset from the middle of the image, need to roll the image to keep the same offset
        offset = self.endcap_left - (self.image_width - self.endcap_right)
        data_dict["data"][self.data_channels] = np.roll(data_dict["data"][self.data_channels], offset, 2)
        # Note: Below assumes y-axis is the tank's azimuth axis. True for IWCD and WCTE, not true for SK, HKFD.
        for v in ["positions", "directions", "three_momenta"]:
            if v in data_dict:
                data_dict[v][..., 2] *= -1
        if "angles" in data_dict:
            data_dict["angles"][..., 1] *= -1
        return data_dict

    def vertical_reflection(self, data_dict):
        """Takes CNN input data and truth info and performs vertical flip, permuting mPMT channels where needed."""
        data_dict["data"][self.data_channels] = self.vertical_image_flip(data_dict["data"])[self.data_channels]
        # Note: Below assumes y-axis is the tank's azimuth axis. True for IWCD and WCTE, not true for SK, HKFD.
        for v in ["positions", "directions", "three_momenta"]:
            if v in data_dict:
                data_dict[v][..., 1] *= -1
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
        barrel = data_dict["data"][self.barrel]
        top_endcap = data_dict["data"][self.top_endcap]
        bottom_endcap = data_dict["data"][self.bottom_endcap]
        # Horizontal flip of the left and right halves of barrel, by flipping and then rolling whole barrel
        # If the endcaps are offset from the middle of the image, need to roll the barrel to keep the same offset
        roll = self.image_width//2 + self.endcap_left - (self.image_width - self.endcap_right)
        barrel[self.data_channels] = np.roll(self.horizontal_image_flip(barrel)[self.data_channels], roll, 2)
        # Vertical flip of the top and bottom endcaps
        top_endcap[self.data_channels] = self.vertical_image_flip(top_endcap)[self.data_channels]
        bottom_endcap[self.data_channels] = self.vertical_image_flip(bottom_endcap)[self.data_channels]
        # Note: Below assumes y-axis is the tank's azimuth axis. True for IWCD and WCTE, not true for SK, HKFD.
        for v in ["positions", "directions", "three_momenta"]:
            if v in data_dict:
                data_dict[v][..., 0] *= -1
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
        barrel = data_dict["data"][self.barrel]
        top_endcap = data_dict["data"][self.top_endcap]
        bottom_endcap = data_dict["data"][self.bottom_endcap]
        # Roll the barrel around by half the columns
        barrel[self.data_channels] = np.roll(barrel[self.data_channels], self.image_width // 2, 2)
        # Vertical and horizontal flips of the endcaps
        top_endcap[self.data_channels] = self.rotate_image(top_endcap)[self.data_channels]
        bottom_endcap[self.data_channels] = self.rotate_image(bottom_endcap)[self.data_channels]
        # Note: Below assumes y-axis is the tank's azimuth axis. True for IWCD and WCTE, not true for SK, HKFD.
        for v in ["positions", "directions", "three_momenta"]:
            if v in data_dict:
                data_dict[v][..., (0, 2)] *= -1
        # rotate azimuth angle by pi, keeping values in range [-pi, pi]
        if "angles" in data_dict:
            data_dict["angles"][..., 1] += np.where(data_dict["angles"][..., 1] > 0, -np.pi, np.pi)
        return data_dict

    def random_reflections(self, data_dict):
        """Takes CNN input data and truth information and randomly reflects the detector about each axis."""
        return du.apply_random_transformations([self.horizontal_reflection, self.vertical_reflection, self.rotation180], data_dict)

    def mpmt_padding(self, data_dict):
        """
        Takes CNN input data in event-display-like format and returns the data with part of the barrel duplicated to one
        side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels in the mPMTs permuted, to
        provide two 'views' of the detector in one image.
        """
        # pad the image with half the barrel width
        padded_data = np.pad(data_dict["data"], ((0, 0), (0, 0), (0, self.image_width//2)), mode="edge")
        # copy the left half of the barrel to the right hand side
        left_barrel = np.array_split(data_dict["data"][self.barrel], 2, axis=2)[0]
        padded_data[:, self.endcap_size:-self.endcap_size, -self.image_width//2:] = left_barrel
        # copy 180-deg rotated end-caps to the appropriate place
        endcap_copy_left = (self.image_width // 2) + self.endcap_left
        endcap_copy_columns = np.s_[endcap_copy_left:endcap_copy_left + self.endcap_size]
        padded_data[:, :self.endcap_size, endcap_copy_columns] = self.rotate_image(data_dict["data"][self.top_endcap])
        padded_data[:, -self.endcap_size:, endcap_copy_columns] = self.rotate_image(data_dict["data"][self.bottom_endcap])
        data_dict["data"] = padded_data
        return data_dict

    def double_cover(self, data_dict):
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
        # Make copies of the endcaps, flipped, to use later
        top_endcap_copy = self.rotate_image(data_dict["data"][self.top_endcap])
        bottom_endcap_copy = self.rotate_image(data_dict["data"][self.bottom_endcap])
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
        barrel_bottom_flipped, barrel_top_flipped = np.array_split(self.rotate_image(barrel_rolled), 2, axis=1)
        data_dict["data"] = np.concatenate((barrel_top_flipped, data, barrel_bottom_flipped), axis=1)
        return data_dict


def collapse_channel(hit_data):
    """Replaces 19 channels for the 19 PMTs within mPMT with two channels corresponding to their mean and stdev."""
    return np.stack((np.mean(hit_data, axis=-3), np.std(hit_data, axis=-3)))
