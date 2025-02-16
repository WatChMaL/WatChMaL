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
            self.dataset_length = h5_file["labels"].shape[0]

        self.label_set = None
        self.labels_key = None
        self.use_memmap = use_memmap

        self.initialized = False

    def initialize(self):
        """
        Initialises the arrays from the HDF5 file. For DistributedDataParallel, this cannot be done when first creating
        the dataset object, since the memmap objects used cannot be pickled and shared amongst distributed data loaders,
        so in that case this function should instead be called only once first actually loading some data.
        """
        self.h5_file = h5py.File(self.h5_path, "r")

        # self.event_ids  = np.array(self.h5_file["event_ids"])
        # self.root_files = np.array(self.h5_file["root_files"])
        self.labels = np.array(self.h5_file["labels"])
        self.positions  = np.array(self.h5_file["positions"])
        self.angles     = np.array(self.h5_file["angles"])
        self.energies   = np.array(self.h5_file["energies"])
        # if "veto" in self.h5_file.keys():
        #     self.veto  = np.array(self.h5_file["veto"])
        #     self.veto2 = np.array(self.h5_file["veto2"])
        self.event_hits_index = np.append(self.h5_file["event_hits_index"], self.h5_file["hit_pmt"].shape[0]).astype(np.int64)

        self.hit_pmt = self.load_hits("hit_pmt")
        self.hit_time = self.load_hits("hit_time")

        # Set attribute so that method won't be invoked again
        self.initialized = True

        # perform label mapping now that labels have been initialised
        if self.label_set is not None:
            self.map_labels(self.label_set, self.labels_key)

    def map_labels(self, label_set, labels_key="labels"):
        """
        Maps the labels of the dataset into a range of integers from 0 up to N-1, where N is the number of unique labels
        in the provided label set.

        Parameters
        ----------
        label_set: sequence of labels
            Set of all possible labels to map onto the range of integers from 0 to N-1, where N is the number of unique
            labels.
        labels_key: string
            Name of the key used for the labels
        """
        self.label_set = set(label_set)
        self.labels_key = labels_key
        if self.initialized:
            try:
                self.original_labels = getattr(self, labels_key)
            except AttributeError:
                self.original_labels = np.array(self.h5_file[labels_key])
            labels = np.ndarray(self.original_labels.shape, dtype=int)
            for i, l in enumerate(self.label_set):
                labels[self.original_labels == l] = i
            setattr(self, labels_key, labels)

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

        data_dict = {
            "labels": self.labels[item].astype(np.int64),
            "energies": self.energies[item].copy(),
            "angles": self.angles[item].copy(),
            "positions": self.positions[item].copy(),
            "directions": direction_from_angles(self.angles[item]),
            # "event_ids": self.event_ids[item],
            # "root_files": self.root_files[item],
            "indices": item
        }
        if self.labels_key is not None and self.labels_key not in data_dict:
            data_dict[self.labels_key] = getattr(self, self.labels_key)[item].copy()
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
    def __init__(self, h5_path, use_memmap=True):
        H5CommonDataset.__init__(self, h5_path, use_memmap)
        
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
