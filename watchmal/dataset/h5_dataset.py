"""
Class for loading data in h5 format
"""

from watchmal.utils.math import direction_from_angles

# torch imports
from torch.utils.data import Dataset

# generic imports
import h5py
import numpy as np
from abc import ABC, abstractmethod


class H5CommonDataset(Dataset, ABC):
    """
    Base class for loading data from HDF5 files containing PMT hit data.
    Sets up access to all of the data that is common between both digitized hit data and truth hits data.
    These are:

    ====================================================================================================================
    Array name        Shape           Data type  Description
    ====================================================================================================================
    event_ids         (n_events,)     int32      ID of the event in the ROOT file
    root_files        (n_events,)     object     File name and location of the ROOT file
    labels            (n_events,)     int32      Label for event classification (gamma=0, electron=1, muon=2)
    positions         (n_events,1,3)  float32    Initial (x, y, z) position of simulated particle
    angles            (n_events,2)    float32    Initial direction of simulated particle as (polar, azimuth) angles
    energies          (n_events,1)    float32    Initial total energy of simulated particle
    veto              (n_events,)     bool       OD veto estimate based on any Cherenkov producing particles exiting the
                                                 tank, with initial energy above threshold
    veto2             (n_events,)     bool       OD veto estimate based on any Cherenkov producing particles exiting the
                                                 tank, with an estimate of energy at the point the particle exits the
                                                 tank being above threshold
    event_hits_index  (n_events,)     int64      Starting index in the hit dataset objects for hits of a particular
                                                 event
    hit_pmt           (n_hits,)       int32      PMT ID of the hit
    hit_time          (n_hits,)       float32    Time of the hit
    ====================================================================================================================
    """
    def __init__(self, h5_path, use_memmap=True):
        """Dataset from the HDF5 file in `h5_path`, using memmaps for hit arrays unless `use_memmap` is set to False"""
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_length = h5_file["event_hits_index"].shape[0]

        self.label_set = None
        self.labels_key = None
        self.use_memmap = use_memmap

        self.initialized = False

        self.h5_file = None
        self.event_hits_index = None
        self.hit_pmt = None
        self.hit_time = None
        self.target_key = None
        self.targets = None
        self.unmapped_labels = None

    def set_target(self, target_key):
        self.target_key = target_key
        if self.target_key is None:
            self.targets = {}
            return
        if self.initialized:
            try:
                if isinstance(self.target_key, str):
                    self.targets = {target_key: self.load_target(target_key)}
                else:
                    self.targets = {t: self.load_target(t) for t in self.target_key}
            except KeyError:
                # truth info don't exist, can only predict but not train or evaluate
                self.targets = {}

    def load_target(self, target_key):
        if target_key == "directions":
            return direction_from_angles(np.array(self.h5_file["angles"]))
        else:
            return np.array(self.h5_file[target_key]).squeeze()

    def initialize(self):
        """
        Initialises the arrays from the HDF5 file. For DistributedDataParallel, this cannot be done when first creating
        the dataset object, since the memmap objects used cannot be pickled and shared amongst distributed data loaders,
        so in that case this function should instead be called only once first actually loading some data.
        """
        self.h5_file = h5py.File(self.h5_path, "r")

        self.event_hits_index = np.append(self.h5_file["event_hits_index"], self.h5_file["hit_pmt"].shape[0]).astype(np.int64)
        self.hit_pmt = self.load_hits("hit_pmt")
        self.hit_time = self.load_hits("hit_time")

        # Set attribute so that method won't be invoked again
        self.initialized = True

        # load targets and perform label mapping now the dataset has been initialised
        self.set_target(self.target_key)
        if self.label_set is not None:
            self.map_labels(self.label_set)

    def map_labels(self, label_set):
        """
        Maps the labels of the dataset into a range of integers from 0 up to N-1, where N is the number of unique labels
        in the provided label set.

        Parameters
        ----------
        label_set: sequence of labels
            Set of all possible labels to map onto the range of integers from 0 to N-1, where N is the number of unique
            labels.
        """
        self.label_set = set(label_set)
        if self.initialized and self.targets:
            self.unmapped_labels = self.targets[self.target_key]
            labels = np.ndarray(self.unmapped_labels.shape, dtype=np.int64)
            for i, l in enumerate(self.label_set):
                labels[self.unmapped_labels == l] = i
            self.targets[self.target_key] = labels

    def load_hits(self, h5_key):
        """Loads data from a given key in the h5 file either into numpy arrays or memmaps"""
        if self.use_memmap:
            data = self.h5_file[h5_key]
            return np.memmap(self.h5_path, mode="r", shape=data.shape, offset=data.id.get_offset(), dtype=data.dtype)
        else:
            return np.array(self.h5_file[h5_key])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()
        data_dict = {k: t[item].copy() for k, t in self.targets.items()}
        data_dict["indices"] = item
        return data_dict


class H5Dataset(H5CommonDataset, ABC):
    """
    Class for loading from an HDF5 file containing digitized hit data. The base class H5CommonDataset handles loading
    data that is common to digitized and true hit datasets, while this class handles loading the additional array that
    only the digitized hit datasets contain:

    =============================================================
    Array name  Shape      Data type  Description
    =============================================================
    hit_charge  (n_hits,)  float32    Charge of the digitized hit
    =============================================================
    """
    def __init__(self, h5_path, use_memmap=True, mask_pmts=None):
        H5CommonDataset.__init__(self, h5_path, use_memmap)
        self.mask_pmts = mask_pmts

    def initialize(self):
        """Creates a memmap for the digitized hit charge data."""
        super().initialize()
        self.hit_charge = self.load_hits("hit_charge")
        
    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        self.event_hit_pmts = self.hit_pmt[start:stop]
        self.event_hit_charges = self.hit_charge[start:stop]
        self.event_hit_times = self.hit_time[start:stop]
        if self.mask_pmts is not None:
            mask = np.isin(self.event_hit_pmts, self.mask_pmts, invert=True)
            self.event_hit_pmts = self.event_hit_pmts[mask]
            self.event_hit_charges = self.event_hit_charges[mask]
            self.event_hit_times = self.event_hit_times[mask]

        return data_dict


class H5TrueDataset(H5CommonDataset, ABC):
    """
    Class for loading from an HDF5 file containing true hit data. The base class H5CommonDataset handles loading data
    that is common to digitised and true hit datasets, while this class handles loading the additional array that only
    the true hit datasets contain:

    ====================================================================================================================
    Array name  Shape      Data type  Description
    ====================================================================================================================
    hit_parent  (n_hits,)  float32    Parent track ID of the true hit, as defined by WCSim's true hit parent. -1 is used
                                      for dark noise.
    ====================================================================================================================
    """

    def __init__(self, h5_path, digitize_hits=True, use_memmap=True):
        H5CommonDataset.__init__(self, h5_path, use_memmap)
        self.digitize_hits = digitize_hits

    def initialize(self):
        """Creates a memmap for the true hit parent data."""
        super().initialize()
        self.hit_parent = self.load_hits("hit_parent")

    def digitize(self, truepmts, truetimes, trueparents):
        """
        Function to perform simulated digitization of true hit information.
        Replace below with a real digitization.  For now take time closest to zero as time, and sum of photons as charge.
        """
        pmt_time_dict = {pmt: truetimes[truepmts == pmt] for pmt in truepmts}
        pmt_photons_dict = {pmt: len(truetimes[truepmts == pmt]) for pmt in truepmts}
        pmt_mintimes_dict = {pmt: min(abs(truetimes[truepmts == pmt])) for pmt in truepmts}

        timeoffset = 950.0
        allpmts = np.array(list(pmt_photons_dict.keys()))
        alltimes = np.array(list(pmt_mintimes_dict.values())) + timeoffset
        allcharges = np.array(list(pmt_photons_dict.values()))
        return allpmts, alltimes, allcharges

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        true_pmts = self.hit_pmt[start:stop].astype(np.int16)
        true_times = self.hit_time[start:stop]
        true_parents = self.hit_parent[start:stop]

        if self.digitize_hits:
            self.event_hit_pmts, self.event_hit_times, self.event_hit_charges = self.digitize(true_pmts, true_times, true_parents)
        else:
            self.event_hit_pmts = true_pmts
            self.event_hit_times = true_times
            self.event_hit_parents = true_parents

        return data_dict
