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

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19


class CNNmPMTDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit charge of the 19 PMTs within each mPMT, and the second
    and third dimensions are the height and width of the CNN image. Each pixel of the image corresponds to one mPMT,
    with mPMTs arrange in an event-display-like format.
    """

    def __init__(self, h5file, mpmt_positions_file, padding_type=None, transforms=None, mode=['charge','time'], collapse_mode=None, scaling_charge=None, scaling_time=None):
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
	mode: sequence of string
	    List defines the PMT data included in the image-like CNN arrays. It can be either 'charge', 'time' or both (default)
        collapse_mode: sequence of string
	    List of the data to be collapsed to two channels, containing, respectively, the mean and the std of other channels. 
	    i.e. provides the mean and the std of PMT charges and/or time in each mPMT instead of providing all PMT data.	
	    It can be [], ['charge'], ['time'] or ['charge', 'time']. By default no collapsing is performed.
        scaling_charge:[offset, scale]
	    Offset and scale to standardise the PMT charge data
        scaling_time: [offset, scale]
	    Offset and scale to standardise the PMT time data
        ----------
	"""
	
        super().__init__(h5file)

        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.data_size = np.max(self.mpmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.mpmt_positions[:, 0] == row) == self.data_size[1]]
        n_channels = pmts_per_mpmt
        self.data_size = np.insert(self.data_size, 0, n_channels)
        self.transforms = du.get_transformations(self, transforms)

        if padding_type is not None:
            self.padding_type = getattr(self, padding_type)
        else:
            self.padding_type = None

        self.horizontal_flip_mpmt_map = [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map = [6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]

        self.mode = mode
        self.collapse_mode = collapse_mode
        self.scaling_charge = scaling_charge
        self.scaling_time = scaling_time

        if self.scaling_charge is not None:
            self.charge_offset = self.scaling_charge[0]
            self.charge_scale = self.scaling_charge[1]

        if self.scaling_time is not None:
            self.time_offset = self.scaling_time[0]
            self.time_scale = self.scaling_time[1] 
            

    def process_data(self, hit_pmts, hit_data):
        """
        Returns event data from dataset associated with a specific index

        Parameters
        ----------
        hit_pmts: array_like of int
            Array of hit PMT IDs
        hit_data: array_like of float
            Array of PMT hit charges, or other per-PMT data

        Returns
        -------
        data: ndarray
            Array in image-like format (channels, rows, columns) for input to CNN network.
        """
        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

        # fix barrel array indexing to match endcaps in xyz ordering
        barrel_data = data[:, self.barrel_rows, :]
        data[:, self.barrel_rows, :] = barrel_data[barrel_map_array_idxs, :, :]

        return data

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)

        if 'charge' in self.mode:
            hit_data = self.event_hit_charges
            if self.scaling_charge is not None:
                hit_data = self.feature_scaling_std(hit_data, self.charge_offset, self.charge_scale)
            
            charge_image = from_numpy(self.process_data(self.event_hit_pmts, hit_data))
            charge_image = self.padding_type(charge_image)
        
        if 'time' in self.mode:
            hit_data = self.event_hit_times
            if self.scaling_time is not None:
                hit_data = self.feature_scaling_std(hit_data, self.time_offset, self.time_scale)

            time_image = from_numpy(self.process_data(self.event_hit_pmts, hit_data))
            time_image = self.padding_type(time_image)

        # Merge all channels
        if ('time' in self.mode) and ('charge' in self.mode):
            processed_image = torch.cat((charge_image, time_image), 0)
	    processed_image = du.apply_random_transformations(self.transforms, processed_image)

            charge_image = processed_image[:19, :, :]
	    time_image = processed_image[19:, :, :]
	    if 'charge' in self.collapse_mode:
        	 mean_channel = torch.mean(charge_image, 0, keepdim=True)
                 std_channel = torch.std(charge_image, 0, keepdim=True)
                 charge_image = torch.cat((mean_channel, std_channel), 0)
	    if 'time' in self.collapse_mode:
		 mean_channel = torch.mean(time_image, 0, keepdim=True)
                 std_channel = torch.std(time_image, 0, keepdim=True)
                 time_image = torch.cat((mean_channel, std_channel), 0)

	    processed_image = torch.cat((charge_image, time_image), 0)

        elif 'charge' in self.mode:
            processed_image = du.apply_random_transformations(self.transforms, charge_image)
	    if 'charge' in self.collapse_mode:
	        mean_channel = torch.mean(processed_image, 0, keepdim=True)
                std_channel = torch.std(processed_image, 0, keepdim=True)
                processed_image = torch.cat((mean_channel, std_channel), 0)
        else:
            processed_image = du.apply_random_transformations(self.transforms, time_image)
	    if 'time' in self.collapse_mode:
                 mean_channel = torch.mean(processed_image, 0, keepdim=True)
                 std_channel = torch.std(processed_image, 0, keepdim=True)
                 processed_image = torch.cat((mean_channel, std_channel), 0)

        data_dict["data"] = processed_image
        
        return data_dict
        
    def horizontal_flip(self, data):
        """
        Takes image-like data and returns the data after applying a horizontal flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
 	flip_mpmt_map = np.tile(self.horizontal_flip_mpmt_map, data.shape[0]/19)
        return flip(data[flip_mpmt_map, :, :], [2])

    def vertical_flip(self, data):
        """
        Takes image-like data and returns the data after applying a vertical flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
  	flip_mpmt_map = np.tile(self.vertical_flip_mpmt_map, data.shape[0]/19)
        return flip(data[flip_mpmt_map, :, :], [1])

    def flip_180(self, data):
        """
        Takes image-like data and returns the data after applying both a horizontal flip to the image. This is
        equivalent to a 180-degree rotation of the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        return self.horizontal_flip(self.vertical_flip(data))
 
    def front_back_reflection(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal flip of the left and
        right halves of the barrels and vertical flip of the endcaps. This is equivalent to reflecting the detector
        swapping the front and back of the event-display view. The channels of the PMTs within mPMTs also have the
        appropriate permutation applied.
        """

        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                     # 5
        half_barrel_width = data.shape[2]//2                    # 20
        l_endcap_index = half_barrel_width - radius_endcap      # 15
        r_endcap_index = half_barrel_width + radius_endcap      # 25
        
        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :half_barrel_width]
        right_barrel = data[:, self.barrel_rows, half_barrel_width:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :half_barrel_width] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, half_barrel_width:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data

    def rotation180(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal and vertical flip of the
        endcaps and shifting of the barrel rows by half the width. This is equivalent to a 180-degree rotation of the
        detector about its axis. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25   

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:, self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data
    
    def mpmt_padding(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with part of the barrel duplicated to one
        side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels in the mPMTs permuted, to
        provide two 'views' of the detect in one image.
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index+1]

        padded_data[:, :barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data

    def double_cover(self, data):
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
        w = data.shape[2]                                                                            
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2
        half_barrel_width, quarter_barrel_width = w//2, w//4

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180 
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width
        r1_endcap_index = l1_endcap_index + 2*radius_endcap
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1:, l1_endcap_index:r1_endcap_index]
        
        padded_data[:, :barrel_row_start, l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)
        
        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]
        
        concat_order = (self.flip_180(barrel_top_half), 
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data
	
    def feature_scaling_std(self, hit_array, offset, scale):
        """
            Scale data using standarization.
        """
        standarized_array = (hit_array - offset)/scale
        return standarized_array
