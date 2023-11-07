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
import watchmal.dataset.data_utils as du

PMTS_PER_MPMT = 19
# maps to permute the PMTs within mPMT for various transformations, etc.
BARREL_MPMT_MAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18])
VERTICAL_FLIP_MPMT_MAP = np.array([6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18])
HORIZONTAL_FLIP_MPMT_MAP = np.array([0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18])


class CNNmPMTDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit charge of the 19 PMTs within each mPMT, and the second
    and third dimensions are the height and width of the CNN image. Each pixel of the image corresponds to one mPMT,
    with mPMTs arrange in an event-display-like format.
    """

    def __init__(self, h5file, mpmt_positions_file, padding_type=None, transforms=None, channels=None,
                 collapse_mpmt_channels=None, channel_scaling=None):
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
        channels: sequence of string
            List defines the PMT data included in the image-like CNN arrays. It can be either 'charge', 'time' or both
            (default)
        collapse_mpmt_channels: sequence of string
            List of the data to be collapsed to two channels, containing, respectively, the mean and the std of other
            channels. i.e. provides the mean and the std of PMT charges and/or time in each mPMT instead of providing
            all PMT data. It can be [], ['charge'], ['time'] or ['charge', 'time']. By default, no collapsing is
            performed.
        channel_scaling: dict of (int, int)
            Dictionary with keys corresponding to channels and values contain the offset and scale to use. By default,
            no scaling is applied.
"""

        super().__init__(h5file)

        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.padding_type = None
        if padding_type is not None:
            self.padding_type = getattr(self, padding_type)
        self.transforms = du.get_transformations(self, transforms)
        if channels is None:
            channels = ['charge', 'time']
        if collapse_mpmt_channels is None:
            collapse_mpmt_channels = []
        self.collapse_channels = collapse_mpmt_channels
        if channel_scaling is None:
            channel_scaling = {}
        self.scaling = channel_scaling

        self.image_height, self.image_width = np.max(self.mpmt_positions, axis=0) + 1
        self.image_depth = 0
        self.channel_ranges = {}
        self.flip_permutation = {'v': np.array([], dtype=int), 'h': np.array([], dtype=int)}
        for c in channels:
            channel_depth = 2 if c in collapse_mpmt_channels else 19 if c in ("charge", "time") else 1
            self.channel_ranges[c] = range(self.image_depth, self.image_depth+channel_depth)
            # permutation maps are needed for applying transformations to the image that affect mPMT channel ordering
            if channel_depth == PMTS_PER_MPMT:
                self.flip_permutation['v'] = np.append(self.flip_permutation['v'], VERTICAL_FLIP_MPMT_MAP + self.image_depth)
                self.flip_permutation['h'] = np.append(self.flip_permutation['h'], HORIZONTAL_FLIP_MPMT_MAP + self.image_depth)
            else:
                self.flip_permutation['v'] = np.append(self.flip_permutation['v'], self.channel_ranges[c])
                self.flip_permutation['h'] = np.append(self.flip_permutation['h'], self.channel_ranges[c])
            self.image_depth += channel_depth
        self.flip_permutation['b'] = self.flip_permutation['h'][self.flip_permutation['v']]

        # make some index expressions for different parts of the image, to use in transformations etc
        rows, counts = np.unique(self.mpmt_positions[:, 0], return_counts=True)  # count occurrences of each row
        # barrel rows are those where the row appears in mpmt_positions as many times as the image width
        barrel_rows = [row for row, count in zip(rows, counts) if count == self.image_width]
        # endcap size is the number of rows before the first barrel row
        self.endcap_size = min(barrel_rows)
        self.barrel = np.s_[..., self.endcap_size:max(barrel_rows) + 1, :]
        # endcaps are assumed to be within squares centred above and below the barrel
        endcap_left = (self.image_width - self.endcap_size) // 2
        endcap_right = endcap_left + self.endcap_size
        self.top_endcap = np.s_[..., :self.endcap_size, endcap_left:endcap_right]
        self.bottom_endcap = np.s_[..., -self.endcap_size:, endcap_left:endcap_right]

    def process_data(self, hit_pmts, hit_data):
        """Returns image-like event data array (channels, rows, columns) from arrays of PMT IDs and data at the PMTs."""
        hit_mpmts = hit_pmts // PMTS_PER_MPMT
        hit_channel = hit_pmts % PMTS_PER_MPMT

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros((PMTS_PER_MPMT, self.image_height, self.image_width), dtype=np.float32)
        data[hit_channel, hit_rows, hit_cols] = hit_data

        # fix indexing of barrel PMTs in mPMT modules to match that of endcaps in the projection to 2D
        data[self.barrel] = data[BARREL_MPMT_MAP][self.barrel]

        return data

    def __getitem__(self, item):
        """Returns image-like event data array (channels, rows, columns) for an event at a given index"""
        data_dict = super().__getitem__(item)
        hit_data = {"charge": self.event_hit_charges, "time": self.event_hit_times}
        # apply scaling to channels
        for c, (offset, scale) in self.scaling.items():
            hit_data[c] = (hit_data[c] - offset)/scale
        # Process the channels
        data = torch.zeros((self.image_depth, self.image_height, self.image_width))
        for c, r in self.channel_ranges.items():
            channel_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data[c]))
            if c in self.collapse_channels:
                channel_data = collapse_channel(channel_data)
            data[r] = channel_data
        # Apply random transformations for augmentation
        data = du.apply_random_transformations(self.transforms, data)
        # Apply "padding" transformation e.g. for double cover
        if self.padding_type is not None:
            data = self.padding_type(data)
        data_dict["data"] = data
        return data_dict

    def image_flip(self, data, direction):
        """
        Takes image-like data and returns the data after applying a horizontal flip to the image.
        The channels of the PMTs within mPMTs are permuted by to flip_mpmt_map.

        Parameters
        ----------
        data: ndarray
            Array in image-like format (channels, rows, columns) for input to CNN network.
        direction: ['v', 'h', 'b']
            The direction to flip the data, 'h' for horizontal, 'v' for vertical, or 'b' for a 180-deg rotation
        """
        dims = {'v': [1], 'h': [2], 'b': [1, 2]}
        channel_permutation = self.flip_permutation[direction] if data.shape[0] == PMTS_PER_MPMT else slice(None)
        return flip(data[channel_permutation, :, :], dims[direction])

    def horizontal_flip(self, data):
        """Perform horizontal flip of detector to each channel, permuting mPMT channels where needed"""
        return self.image_flip(data, 'h')

    def vertical_flip(self, data):
        """Perform vertical flip of detector to each channel, permuting mPMT channels where needed"""
        return self.image_flip(data, 'v')

    def front_back_reflection(self, data):
        """
        Takes a dictionary of CNN input data channels in event-display-like format and returns the data with horizontal
        flip of the left and right halves of the barrels and vertical flip of the end-caps. This is equivalent to
        reflecting the detector swapping the front and back of the event-display view. The channels of the PMTs within
        mPMTs also have the appropriate permutation applied.
        """
        # Horizontal flip of the left and right halves of barrel
        left_barrel, right_barrel = torch.tensor_split(data[self.barrel], 2, dim=2)
        left_barrel[:] = self.image_flip(left_barrel, 'h')
        right_barrel[:] = self.image_flip(right_barrel, 'h')
        # Vertical flip of the top and bottom endcaps
        data[self.top_endcap] = self.image_flip(data[self.top_endcap], 'v')
        data[self.bottom_endcap] = self.image_flip(data[self.bottom_endcap], 'v')
        return data

    def rotation180(self, data):
        """
        Takes a dictionary of CNN input data channels in event-display-like format and returns the data with horizontal
        and vertical flip of the endcaps and shifting of the barrel rows by half the width. This is equivalent to a
        180-degree rotation of the detector about its axis. The channels of the PMTs within mPMTs also have the
        appropriate permutation applied.
        """
        # Vertical and horizontal flips of the endcaps
        data[self.top_endcap] = self.image_flip(self.top_endcap, 'b')
        data[self.bottom_endcap] = self.image_flip(self.bottom_endcap, 'b')
        # Roll the barrel around by half the columns
        data[self.barrel] = torch.roll(data[self.barrel], self.image_width // 2, 2)
        return data

    def mpmt_padding(self, data):
        """
        Takes a dictionary of CNN input data channels in event-display-like format and returns the data with part of the
        barrel duplicated to one side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels
        in the mPMTs permuted, to provide two 'views' of the detector in one image.
        """
        # copy the left half of the barrel (padded with zeros above and below) to the right hand side
        left_barrel = torch.tensor_split(data[self.barrel], 2, dim=2)[0]
        left_barrel_padded = torch.nn.functional.pad(left_barrel, (self.endcap_size, self.endcap_size, 0, 0))
        padded_data = torch.cat((data, left_barrel_padded), dim=2)
        # copy 180-deg rotated end-caps to the appropriate place
        endcap_copy_left = data.shape[2] - (self.endcap_size // 2)
        endcap_copy_right = endcap_copy_left + self.endcap_size
        padded_data[:, :self.endcap_size, endcap_copy_left:endcap_copy_right] = self.image_flip(data[self.top_endcap], 'b')
        padded_data[:, -self.endcap_size:, endcap_copy_left:endcap_copy_right] = self.image_flip(data[self.bottom_endcap], 'b')
        return padded_data

    def double_cover(self, data):
        """
        Takes a dictionary of CNN input data channels in event-display-like format and returns the data with all parts
        of the detector duplicated and rearranged to provide a double-cover of the image, providing two 'views' of the
        detector from a single image with less blank space and physically meaningful cyclic boundary conditions at the
        edges of the image.

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
        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        quarter_barrel_width = self.image_width // 4
        padded_data = torch.roll(data, -quarter_barrel_width, 2)
        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180
        endcap_copy_left = quarter_barrel_width - (self.endcap_size // 2)
        endcap_copy_right = endcap_copy_left + self.endcap_size
        top_endcap_copy = np.s_[..., :self.endcap_size, endcap_copy_left:endcap_copy_right]
        bottom_endcap_copy = np.s_[..., -self.endcap_size:, endcap_copy_left:endcap_copy_right]
        padded_data[top_endcap_copy] = self.image_flip(data[self.top_endcap], 'b')
        padded_data[bottom_endcap_copy] = self.image_flip(data[self.bottom_endcap], 'b')
        # Step - 3 : Rotate the bottom and top halves of barrel and concatenate to the top and bottom of the image
        barrel_flipped = self.image_flip(data[self.barrel], 'b')
        barrel_bottom_flipped, barrel_top_flipped = torch.tensor_split(barrel_flipped, 2, dim=1)
        padded_data = torch.cat((barrel_bottom_flipped, padded_data, barrel_top_flipped), dim=1)
        return padded_data


def collapse_channel(hit_data):
    """
    Replaces 19 channels for the 19 PMTs within mPMT with two channels corresponding to their mean and stdev.
    """
    mean_channel = torch.mean(hit_data, 0, keepdim=True)
    std_channel = torch.std(hit_data, 0, keepdim=True)
    hit_data = torch.cat((mean_channel, std_channel), 0)
    return hit_data
