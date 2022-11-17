"""
Class implementing a dataset for pointnet in h5 format
"""

# generic imports
import numpy as np

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du


class PointNetDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, use_times=True, use_orientations=False, n_points=4000, transforms=None):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge and/or time
        data is formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to
        one PMT and the channels correspond to charge and/or time at each PMT.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        geometry_file: string
            Location of an npz file containing the position and orientation of PMTs
        use_times: bool
            Whether to use PMT hit times as one of the initial PointNet channels. True by default.
        use_orientations: bool
            Whether to use PMT orientation as some of the initial PointNet channels. False by default.
        n_points: int
            Number of points to pass to the PointNet network. If there are fewer hits in an event than `n_points`, then
            additional points are added filled with zeros. If there are more hits in an event than `n_points`, then the
            hit data is truncated and only the first `n_points` hits are passed to the network.
        transforms
            List of random transforms to apply to data before passing to CNN for data augmentation. Each element of the
            list should be the name of a function in watchmal.dataset.pointnet.transformations that performs the
            transformation.
        """
        super().__init__(h5file)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = geo_file["position"].astype(np.float32)
        self.geo_orientations = geo_file["orientation"].astype(np.float32)
        self.use_orientations = use_orientations
        self.use_times = use_times
        self.n_points = n_points
        self.transforms = du.get_transformations(transformations, transforms)
        self.channels = 4
        if use_orientations:
            self.channels += 3
        if use_times:
            self.channels += 1

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)

        n_hits = min(self.n_points, self.event_hit_pmts.shape[0])
        hit_positions = self.geo_positions[self.event_hit_pmts[:n_hits], :]
        data = np.zeros((self.channels, self.n_points), dtype=np.float32)
        data[:3, :n_hits] = hit_positions.T
        if self.use_orientations:
            hit_orientations = self.geo_orientations[self.event_hit_pmts[:n_hits], :]
            data[3:6, :n_hits] = hit_orientations.T
        if self.use_times:
            data[-2, :n_hits] = self.event_hit_times[:n_hits]
        data[-1, :n_hits] = self.event_hit_charges[:n_hits]

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict