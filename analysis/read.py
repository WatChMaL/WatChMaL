import uproot
import glob
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from os.path import dirname
from omegaconf import OmegaConf
from matplotlib import pyplot as plt


class WatChMaLOutput(ABC, metaclass=ABCMeta):
    """Base class for reading in results of a WatChMaL run."""
    def __init__(self, directory, indices=None):
        """
        Create an object holding results of a WatChMaL run, given the run output directory

        Parameters
        ----------
        directory: str
            path to the run's output directory
        indices: np.ndarray of int, optional
            array of indices to specify which events to use when loading outputs, out of the indices output by WatChMaL
            (by default use all events sorted by their indices).
        """
        self.directory = directory
        self.indices = indices
        self._training_log = None
        self._log_train = None
        self._train_log_epoch = None
        self._train_log_loss = None
        self._log_val = None
        self._val_log_epoch = None
        self._val_log_loss = None
        self._val_log_best = None

    def plot_training_progression(self, plot_best=True, y_lim=None, fig_size=None, title=None, legend='center right'):
        """
        Plot the progression of training and validation loss from the run's logs

        Parameters
        ----------
        plot_best: bool, optional
            If true (default), plot points indicating the best validation loss
        y_lim: (int, int), optional
            Range for the y-axis (loss). By default, the range will expand to show all loss values in the logs.
        fig_size: (float, float), optional
            Size of the figure
        title: str, optional
            Title of the figure. By default, do not plot a title.
        legend: str, optional
            Position to plot the legend. By default, the legend is placed in the center right. For no legend use `None`.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(title)
        ax.plot(self.train_log_epoch, self.train_log_loss, lw=2, label='Train loss', color='b', alpha=0.3)
        ax.plot(self.val_log_epoch, self.val_log_loss, lw=2, label='Validation loss', color='b')
        if plot_best:
            ax.plot(self.val_log_epoch[self.val_log_best], self.val_log_loss[self.val_log_best], lw=0, marker='o',
                    label='Best validation loss', color='darkblue')
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.set_ylabel("Loss", c='b')
        ax.set_xlabel("Epoch")
        if legend:
            ax.legend(loc=legend)
        return fig, ax

    def read_training_log(self):
        """
        Read the training progression logs for the run. If the run does not have a training progression log, then logs
        are loaded from a run directory corresponding to a loaded pre-trained state.

        Returns
        -------
        tuple
            Tuple of arrays of training progression log values, see `read_training_log_from_csv` for details.
        """
        train_files = glob.glob(self.directory + "/log_train*.csv")
        if train_files:
            return self.read_training_log_from_csv(self.directory)
        else:  # search for a previous training run with a saved state that was loaded
            conf = OmegaConf.load(self.directory + '/.hydra/config.yaml')
            state_file = conf.tasks.restore_state.weight_file
            directory = dirname(dirname(state_file))
            return self.read_training_log_from_csv(directory)

    def get_outputs(self, name):
        """
        Read the outputs resulting from the evaluation run of a WatChMaL model.

        Parameters
        ----------
        name: str
            name of the output to load

        Returns
        -------
        np.ndarray
            Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
            contains the softmax values of a class.
        """
        outputs = np.load(self.directory + "/" + name + ".npy")
        output_indices = np.load(self.directory + "/indices.npy")
        if self.indices is None:
            return outputs[output_indices.argsort()].squeeze()
        intersection = np.intersect1d(self.indices, output_indices, return_indices=True)
        sorted_outputs = np.zeros(self.indices.shape + outputs.shape[1:])
        sorted_outputs[intersection[1]] = outputs[intersection[2]]
        return sorted_outputs.squeeze()

    @abstractmethod
    def read_training_log_from_csv(self, directory):
        """
        Read the training progression logs from the given directory.

        Parameters
        ----------
        directory: str
            Path to the directory of the training run.

        Returns
        -------
        tuple
            Tuple of arrays of training progression log values, see `read_training_log_from_csv` for details.
        """
        train_files = glob.glob(directory + "/log_train*.csv")
        self._log_train = np.array([np.genfromtxt(f, delimiter=',', skip_header=1) for f in train_files])
        self._log_val = np.genfromtxt(directory + "/log_val.csv", delimiter=',', skip_header=1)
        train_iteration = self._log_train[0, :, 0]
        train_epoch = self._log_train[0, :, 1]
        it_per_epoch = np.min(train_iteration[train_epoch == 1]) - 1
        self._train_log_epoch = train_iteration / it_per_epoch
        self._train_log_loss = np.mean(self._log_train[:, :, 2], axis=0)
        self._val_log_epoch = self._log_val[:, 0] / it_per_epoch

    @property
    def training_log(self):
        """Tuple of arrays of training progression log values, see `read_training_log_from_csv` for details."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._training_log

    @property
    def train_log_epoch(self):
        """Array of train epoch values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_epoch

    @property
    def train_log_loss(self):
        """Array of train loss values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_loss

    @property
    def val_log_epoch(self):
        """Array of validation epoch values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_epoch

    @property
    def val_log_loss(self):
        """Array of validation loss values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_loss

    @property
    def val_log_best(self):
        """
        Array of boolean values indicating whether each entry had the best validation loss so far in the training
        progression log
        """
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_best


class FiTQunOutput:
    """
    Class for reading in results of fiTQun reconstruction. Documentation of the outputs provided is mostly taken
    directly from the fiTQun readme file. See github.com/fiTQun/fiTQun (access to private repository required) for more
    details.

    Time-window information
    -----------------------
    The following attributes are provided for the fiTQun outputs of the hit time clustering algorithm:

    ======================================================================================================
    Attribute name        fiTQun output     Description
    ======================================================================================================
    n_timewindows         fqntwnd           Number of time windows (good clusters) in this event
    timewindow            fqtwnd            Cluster index of the time window(corresponds to cluster_ncand)
    timewindow_cluster    fqtwnd_iclstr     Number of peaks(sub-events) in the time window
    timewindow_time       fqtwnd_prftt0     Pre-fitter vertex time
    timewindow_position   fqtwnd_prftpos    Pre-fitter vertex position
    timewindow_npeaks     fqtwnd_npeak      Time window start/end time
    timewindow_peak_time  fqtwnd_peakt0     Time of each sub-event in the time window
    timewindow_peakiness  fqtwnd_peakiness  Vertex goodness parameter evaluated at the peak position
    ======================================================================================================

    Sub-event information
    ---------------------
    The following attributes are provided for the fiTQun outputs of the sub-events of the hit time clustering algorithm:

    ===================================================================================================================
    Attribute name               fiTQun output  Description
    ===================================================================================================================
    n_subevents                  fqnse          Total number of subevents in the event
    subevent_timewindow          fqitwnd        Index of the time window to which the subevent belongs
    subevent_peak                fqipeak        Peak index within the time window
    subevent_nhits               fqnhitpmt      Number of hits in each subevent
    subevent_total_charge        fqtotq         Total charge in each subevent
    subevent_0ring_total_charge  fq0rtotmu      Total predicted charge for the 0-ring hypothesis in each subevent -
                                                these variables are the result of evaluating the likelihood function
                                                with a particle that is below Cherenkov threshold.
    subevent_0ring_nll           fq0rnll        -log(L) value for the 0-ring hypothesis in each subevent
    subevent_n50                 fqn50          n50 - In the TOF-corrected hit time distribution, number of hits within
                                                the 50ns time window centred at vertex time (1R electron fit vertex is
                                                used)
    subevent_q50                 fqq50          q50 - Total charge of hits included in n50 above
    ===================================================================================================================

    1-Ring fits
    -----------
    These variables are the result of the 1-ring fits. The first index is the subevent (1-ring fits are run on all
    subevents). The second index is the particle-hypothesis index (same as apfit):
    0 = GAMMA, 1 = ELECTRON, 2 = MUON, 3 = PION, 4 = KAON, 5 = PROTON,  6 = CONE GENERATOR
    Currently, only the electron, muon, and pion (the upstream pion segment) hypotheses are implemented.

    The following attributes are provided for the fiTQun outputs of electron and muon sub-fits of the 1-ring fit results
    for the first sub-event:

    ====================================================================================================================
    Attribute name         fiTQun output         Description
    ====================================================================================================================
    electron_flag          fq1rpcflg [][0][1]    Flag to indicate whether fiTQun believes the electron is exiting the ID
                                                 (<0 if MINUIT did not converge)
    electron_momentum      fq1rmom   [][0][1]    Fit electron momentum
    electron_position      fq1rpos   [][0][1][]  Fit electron vertex (0=X, 1=Y, 2=Z)
    electron_direction     fq1rdir   [][0][1][]  Fit electron direction (0=X, 1=Y, 2=Z)
    electron_time          fq1rt0    [][0][1]    Fit electron creation time
    electron_total_charge  fq1rtotmu [][0][1]    Electron best-fit total predicted charge
    electron_nll           fq1rnll   [][0][1]    Electron best-fit -lnL
    muon_flag              fq1rpcflg [][0][2]    Flag to indicate whether fiTQun believes the muon is exiting the ID
                                                 (<0 if MINUIT did not converge)
    muon_momentum          fq1rmom   [][0][2]    Fit muon momentum
    muon_position          fq1rpos   [][0][2][]  Fit muon vertex (0=X, 1=Y, 2=Z)
    muon_direction         fq1rdir   [][0][2][]  Fit muon direction (0=X, 1=Y, 2=Z)
    muon_time               fq1rt0   [][0][2]    Fit muon creation time
    muon_total_charge      1rtotmu   [][0][2]    Muon best-fit total predicted charge
    muon_nll               fq1rnll   [][0][2]    Muon best-fit -lnL
    ====================================================================================================================

    Pi0 fits
    --------
    Pi0 fits are only run on the first sub-event. Index 0 gives the standard, unconstrained pi0 fit. (Index 1 is not
    filled currently)
    The following attributes are provided for the fiTQun outputs of the unconstrained-mass sub-fit of the pi0 fit for
    the first sub-event:

    ============================================================================================================
    Attribute name               fiTQun output           Description
    ============================================================================================================
    pi0_flag                     fqpi0pcflg     [][0]    (PCflg for photon 1) + 2*(PCflg for photon 2)
    pi0_momentum                 fqpi0momtot    [][0]    Fit momentum of the pi0
    pi0_position                 fqpi0pos       [][0][]  Fit vertex position
    pi0_direction                fqpi0dirtot    [][0][]  Fit direction of the pi0
    pi0_time                     fqpi0t0        [][0]    Fit pi0 creation time
    pi0_total_charge             fqpi0totmu     [][0]    Best fit total predicted charge
    pi0_nll                      fqpi0nll       [][0]    Best fit -log(Likelihood)
    pi0_mass                     fqpi0mass      [][0]    Fit pi0 mass (always 134.9766 for constrained mass fit)
    pi0_gamma1_momentum          fqpi0mom1      [][0]    Fit momentum of first photon
    pi0_gamma2_momentum          fqpi0mom2      [][0]    Fit momentum of second photon
    pi0_gamma1_direction         fqpi0dir2      [][0][]  Fit direction of the first photon
    pi0_gamma2_direction         fqpi0dir2      [][0][]  Fit direction of the second photon
    pi0_gamma1_conversion_length fqpi0dconv2    [][0]    Fit conversion length for the first photon
    pi0_gamma2_conversion_length fqpi0dconv2    [][0]    Fit conversion length for the second photon
    pi0_gamma_opening_angle      fqpi0photangle [][0]    Fit opening angle between the photons
    ============================================================================================================

    Multi-Ring fits
    ---------------
    These are the results of the Multi-Ring (MR) fits. The number of executed multi-ring fits depends on the event
    topology, and the first index specifies different fit results. (Index 0 is the best-fit result.)
    Each fit result is assigned a unique fit ID which tells the type of the fit(see fiTQun.cc for more details):

    8-digit ID "N0...ZYX" :
        These are the raw MR fitter output, in which a ring is either an electron or a pi+. The most significant digit
        "N" is the number of rings(1-6), and X, Y and Z are the particle type(as in 1R fit, "1" for e, "3" for pi+) of
        the 1st, 2nd and 3rd ring respectively. Negative fit ID indicates that the ring which is newly added in the fit
        is determined as a fake ring by the fake ring reduction algorithm.

    9-digit ID "1N0...ZYX" :
        These are the results after the fake ring reduction is applied on the raw MR fit results above with ID
        "N0...ZYX". Rings are re-ordered according to their visible energy, and one needs refer to the fqmrpid variable
        for the particle type of each ring, not the fit ID.

    9-digit ID "2N0...ZYX" :
        These are the results after the fake ring merging and sequential re-fitting are applied on the post-reduction
        result "1N0...ZYX". PID of a ring can change after the re-fit, and muon hypothesis is also applied on the most
        energetic ring.

    9-digit ID "3N0...ZYX" :
        These are the results after simultaneously fitting the longitudinal vertex position and the visible energy of
        all rings, on the post-refit result "2N0...ZYX".(Still experimental)

    9-digit ID "8NX000000" :
        When the best-fit hypothesis has more than one ring, the negative log-likelihood values for each ring (N) and
        PID hypothesis (X) can be obtained using these results. For example, to compare the likelihood for the pion and
        electron hypotheses of the second ring, the IDs "813000000" and "811000000" could be used.

    The following attributes are provided for the fiTQun outputs of the multi-ring fits:

    ====================================================================================================================
    Attribute name               fiTQun output  Description
    ====================================================================================================================
    n_multiring_fits             fqnmrfit       Number of MR fit results that are available
    multiring_fit_id             fqmrifit       Fit ID of each MR fit result
    multiring_n_rings            fqmrnring      Number of rings for this fit [1-6]
    multiring_flag               fqmrpcflg      <0 if MINUIT did not converge during the fit
    multiring_pid                fqmrpid        Particle type index for each ring in the fit (Same convention as 1R fit)
    multiring_momentum           fqmrmom        Fit momentum of each ring
    multiring_position           fqmrpos        Fit vertex position of each ring
    multiring_direction          fqmrdir        Fit direction of each ring
    multiring_time               fqmrt0         Fit creation time of each ring
    multiring_total_charge       fqmrtotmu      Best-fit total predicted charge
    multiring_nll                fqmrnll        Best-fit -lnL
    multiring_conversion_length  fqmrdconv      Fit conversion length of each ring(always "0" in default mode)
    multiring_energy_loss        fqmreloss      Energy lost in the upstream track segment(for upstream tracks only)
    ====================================================================================================================

    Multi-Segment Muon fits
    -----------------------
    These are the results of the Multi-Segment (M-S) muon fits. By default, the stand-alone M-S fit (first index="0") is
    applied on every event, and if the most energetic ring in the best-fit MR fit is a muon, the muon track is re-fitted
    as a M-S track. (first index="1")
    The following attributes are provided for the fiTQun outputs of the M-S fits:

    ====================================================================================================================
    Attribute name             fiTQun output  Description
    ====================================================================================================================
    n_multisegment_fits        fqmsnfit       Number of Multi-Segment fit results that are available
    multisegment_flag          fqmspcflg      <0 if MINUIT did not converge during the fit
    multisegment_n_segments    fqmsnseg       Number of track segments in the fit
    multisegment_pid           fqmspid        Particle type of the M-S track (always "2")
    multisegment_fit_id        fqmsifit       Fit ID of the MR fit that seeded this fit("1" for the stand-alone M-S fit)
    multisegment_ring          fqmsimer       Index of the ring to which the M-S track corresponds in the seeding MR fit
    multisegment_momentum      fqmsmom        Fit initial momentum of each segment
    multisegment_position      fqmspos        Fit vertex position of each segment
    multisegment_direction     fqmsdir        Fit direction of each segment
    multisegment_time          fqmst0         Fit creation time of each segment
    multisegment_total_charge  fqmstotmu      Best-fit total predicted charge
    multisegment_nll           fqmsnll        Best-fit -lnL
    multisegment_energy_loss   fqmseloss      Energy lost in each segment
    ====================================================================================================================

    Proton decay: p -> K+ nu; K+ -> mu+ nu; "prompt gamma method" fit
    -----------------------------------------------------------------
    These are the results of the PDK_MuGamma fit, dedicated to proton decay searches. Although there are two available
    fit results for each quantity, only the first is used (e.g. fqpmgmom1[0])
    The following attributes are provided for the fiTQun outputs of the PDK fit:

    ==========================================================================
    Attribute name       fiTQun output  Description
    ==========================================================================
    pdk_flag             fqpmgpcflg     (PCflg for muon) + 2*(PCflg for gamma)
    pdk_muon_momentum    fqpmgmom1      Best-fit muon momentum
    pdk_muon_position    fqpmgpos1      Best-fit muon position
    pdk_muon_direction   fqpmgdir1      Best-fit muon direction
    pdk_muon_time        fqpmgt01       Best-fit muon time
    pdk_gamma_momentum   fqpmgmom2      Best-fit gamma momentum
    pdk_gamma_position   fqpmgpos2      Best-fit gamma position
    pdk_gamma_direction  fqpmgdir2      Best-fit gamma direction
    pdk_gamma_time       fqpmgt02       Best-fit gamma time
    pdk_total_charge     fqpmgtotmu     Best-fit total predicted charge
    pdk_nll              fqpmgnll       Best-fit negative log-likelihood
    ==========================================================================
    """
    def __init__(self, file_path):
        """
        Create an object holding results of a fiTQun reconstruction run, given path to the output root file.

        Parameters
        ----------
        file_path: str
            Path the fiTQun output root file
        """
        self.chain = uproot.lazy(file_path)

        self.n_timewindows = self.chain['fqntwnd']
        self.timewindow = self.chain['fqtwnd']
        self.timewindow_cluster = self.chain['fqtwnd_iclstr']
        self.timewindow_time = self.chain['fqtwnd_prftt0']
        self.timewindow_position = self.chain['fqtwnd_prftpos']
        self.timewindow_npeaks = self.chain['fqtwnd_npeak']
        self.timewindow_peak_time = self.chain['fqtwnd_peakt0']
        self.timewindow_peakiness = self.chain['fqtwnd_peakiness']

        self.n_subevents = self.chain['fqnse']
        self.subevent_timewindow = self.chain['fqitwnd']
        self.subevent_peak = self.chain['fqipeak']
        self.subevent_nhits = self.chain['fqnhitpmt']
        self.subevent_total_charge = self.chain['fqtotq']
        self.subevent_0ring_total_charge = self.chain['fq0rtotmu']
        self.subevent_0ring_nll = self.chain['fq0rnll']
        self.subevent_n50 = self.chain['fqn50']
        self.subevent_q50 = self.chain['fqq50']

        self._singlering_flag = self.chain['fq1rpcflg']
        self._singlering_momentum = self.chain['fq1rmom']
        self._singlering_position = self.chain['fq1rpos']
        self._singlering_direction = self.chain['fq1rdir']
        self._singlering_time = self.chain['fq1rt0']
        self._singlering_total_charge = self.chain['fq1rtotmu']
        self._singlering_nll = self.chain['fq1rnll']
        self._singlering_conversion_length = self.chain['fq1rdconv']
        self._singlering_energy_loss = self.chain['fq1reloss']

        self._electron_flag = None
        self._electron_momentum = None
        self._electron_position = None
        self._electron_direction = None
        self._electron_time = None
        self._electron_total_charge = None
        self._electron_nll = None

        self._muon_flag = None
        self._muon_momentum = None
        self._muon_position = None
        self._muon_direction = None
        self._muon_time = None
        self._muon_total_charge = None
        self._muon_nll = None

        self._pi0fit_flag = self.chain['fqpi0pcflg']
        self._pi0fit_momentum = self.chain['fqpi0momtot']
        self._pi0fit_position = self.chain['fqpi0pos']
        self._pi0fit_direction = self.chain['fqpi0dirtot']
        self._pi0fit_time = self.chain['fqpi0t0']
        self._pi0fit_total_charge = self.chain['fqpi0totmu']
        self._pi0fit_nll = self.chain['fqpi0nll']
        self._pi0fit_mass = self.chain['fqpi0mass']
        self._pi0fit_gamma1_momentum = self.chain['fqpi0mom1']
        self._pi0fit_gamma2_momentum = self.chain['fqpi0mom2']
        self._pi0fit_gamma1_direction = self.chain['fqpi0dir1']
        self._pi0fit_gamma2_direction = self.chain['fqpi0dir2']
        self._pi0fit_gamma1_conversion_length = self.chain['fqpi0dconv1']
        self._pi0fit_gamma2_conversion_length = self.chain['fqpi0dconv2']
        self._pi0fit_gamma_opening_angle = self.chain['fqpi0photangle']

        self._pi0_flag = None
        self._pi0_momentum = None
        self._pi0_position = None
        self._pi0_direction = None
        self._pi0_time = None
        self._pi0_total_charge = None
        self._pi0_nll = None
        self._pi0_mass = None
        self._pi0_gamma1_momentum = None
        self._pi0_gamma2_momentum = None
        self._pi0_gamma1_direction = None
        self._pi0_gamma2_direction = None
        self._pi0_gamma1_conversion_length = None
        self._pi0_gamma2_conversion_length = None
        self._pi0_gamma_opening_angle = None

        self.n_multiring_fits = self.chain['fqnmrfit']
        self.multiring_fit_id = self.chain['fqmrifit']
        self.multiring_n_rings = self.chain['fqmrnring']
        self.multiring_flag = self.chain['fqmrpcflg']
        self.multiring_pid = self.chain['fqmrpid']
        self.multiring_momentum = self.chain['fqmrmom']
        self.multiring_position = self.chain['fqmrpos']
        self.multiring_direction = self.chain['fqmrdir']
        self.multiring_time = self.chain['fqmrt0']
        self.multiring_total_charge = self.chain['fqmrtotmu']
        self.multiring_nll = self.chain['fqmrnll']
        self.multiring_conversion_length = self.chain['fqmrdconv']
        self.multiring_energy_loss = self.chain['fqmreloss']

        self.n_multisegment_fits = self.chain['fqmsnfit']
        self.multisegment_flag = self.chain['fqmspcflg']
        self.multisegment_n_segments = self.chain['fqmsnseg']
        self.multisegment_pid = self.chain['fqmspid']
        self.multisegment_fit_id = self.chain['fqmsifit']
        self.multisegment_ring = self.chain['fqmsimer']
        self.multisegment_momentum = self.chain['fqmsmom']
        self.multisegment_position = self.chain['fqmspos']
        self.multisegment_direction = self.chain['fqmsdir']
        self.multisegment_time = self.chain['fqmst0']
        self.multisegment_total_charge = self.chain['fqmstotmu']
        self.multisegment_nll = self.chain['fqmsnll']
        self.multisegment_energy_loss = self.chain['fqmseloss']

        self.pdk_flag = self.chain['fqpmgpcflg']
        self.pdk_muon_momentum = self.chain['fqpmgmom1']
        self.pdk_muon_position = self.chain['fqpmgpos1']
        self.pdk_muon_direction = self.chain['fqpmgdir1']
        self.pdk_muon_time = self.chain['fqpmgt01']
        self.pdk_gamma_momentum = self.chain['fqpmgmom2']
        self.pdk_gamma_position = self.chain['fqpmgpos2']
        self.pdk_gamma_direction = self.chain['fqpmgdir2']
        self.pdk_gamma_time = self.chain['fqpmgt02']
        self.pdk_total_charge = self.chain['fqpmgtotmu']
        self.pdk_nll = self.chain['fqpmgnll']

    @property
    def electron_flag(self):
        """Flag to indicate whether fiTQun believes the electron is exiting the ID(<0 if MINUIT did not converge)"""
        if self._electron_flag is None:
            self._electron_flag = self._singlering_flag[:, 0, 1]
        return self._electron_flag

    @property
    def electron_momentum(self):
        """Single electron-like ring fit momentum"""
        if self._electron_momentum is None:
            self._electron_momentum = self._singlering_momentum[:, 0, 1]
        return self._electron_momentum

    @property
    def electron_position(self):
        """Single electron-like ring fit vertex (X, Y, Z)"""
        if self._electron_position is None:
            self._electron_position = self._singlering_position[:, 0, 1, :]
        return self._electron_position

    @property
    def electron_direction(self):
        """Single electron-like ring fit direction (X, Y, Z)"""
        if self._electron_direction is None:
            self._electron_direction = self._singlering_direction[:, 0, 1, :]
        return self._electron_direction

    @property
    def electron_time(self):
        """Single electron-like ring fit particle creation time"""
        if self._electron_time is None:
            self._electron_time = self._singlering_time[:, 0, 1]
        return self._electron_time

    @property
    def electron_total_charge(self):
        """Single electron-like ring best-fit total predicted charge"""
        if self._electron_total_charge is None:
            self._electron_total_charge = self._singlering_total_charge[:, 0, 1]
        return self._electron_total_charge

    @property
    def electron_nll(self):
        """Single electron-like ring best-fit -lnL"""
        if self._electron_nll is None:
            self._electron_nll = self._singlering_nll[:, 0, 1]
        return self._electron_nll

    @property
    def muon_flag(self):
        """Flag to indicate whether fiTQun believes the muon is exiting the ID(<0 if MINUIT did not converge)"""
        if self._muon_flag is None:
            self._muon_flag = self._singlering_flag[:, 0, 2]
        return self._muon_flag

    @property
    def muon_momentum(self):
        """Single muon-like ring fit momentum"""
        if self._muon_momentum is None:
            self._muon_momentum = self._singlering_momentum[:, 0, 2]
        return self._muon_momentum

    @property
    def muon_position(self):
        """Single muon-like ring fit vertex (X, Y, Z)"""
        if self._muon_position is None:
            self._muon_position = self._singlering_position[:, 0, 2, :]
        return self._muon_position

    @property
    def muon_direction(self):
        """Single muon-like ring fit direction (X, Y, Z)"""
        if self._muon_direction is None:
            self._muon_direction = self._singlering_direction[:, 0, 2, :]
        return self._muon_direction

    @property
    def muon_time(self):
        """Single muon-like ring fit particle creation time"""
        if self._muon_time is None:
            self._muon_time = self._singlering_time[:, 0, 2]
        return self._muon_time

    @property
    def muon_total_charge(self):
        """Single muon-like ring best-fit total predicted charge"""
        if self._muon_total_charge is None:
            self._muon_total_charge = self._singlering_total_charge[:, 0, 2]
        return self._muon_total_charge

    @property
    def muon_nll(self):
        """Single muon-like ring best-fit -lnL"""
        if self._muon_nll is None:
            self._muon_nll = self._singlering_nll[:, 0, 2]
        return self._muon_nll

    @property
    def pi0_flag(self):
        """(PCflg for photon 1) + 2*(PCflg for photon 2)"""
        if self._pi0_flag is None:
            self._pi0_flag = self._pi0fit_flag[:, 0]
        return self._pi0_flag

    @property
    def pi0_momentum(self):
        """Fit momentum of the pi0"""
        if self._pi0_momentum is None:
            self._pi0_momentum = self._pi0fit_momentum[:, 0]
        return self._pi0_momentum

    @property
    def pi0_position(self):
        """pi0 fit vertex position"""
        if self._pi0_position is None:
            self._pi0_position = self._pi0fit_position[:, 0, :]
        return self._pi0_position

    @property
    def pi0_direction(self):
        """Fit direction of the pi0"""
        if self._pi0_direction is None:
            self._pi0_direction = self._pi0fit_direction[:, 0, :]
        return self._pi0_direction

    @property
    def pi0_time(self):
        """Fit pi0 creation time"""
        if self._pi0_time is None:
            self._pi0_time = self._pi0fit_time[:, 0]
        return self._pi0_time

    @property
    def pi0_total_charge(self):
        """pi0 best-fit total predicted charge"""
        if self._pi0_total_charge is None:
            self._pi0_total_charge = self._pi0fit_total_charge[:, 0]
        return self._pi0_total_charge

    @property
    def pi0_nll(self):
        """pi0 best-fit -log(Likelihood)"""
        if self._pi0_nll is None:
            self._pi0_nll = self._pi0fit_nll[:, 0]
        return self._pi0_nll

    @property
    def pi0_mass(self):
        """Fit pi0 mass (always 134.9766 for constrained mass fit)"""
        if self._pi0_mass is None:
            self._pi0_mass = self._pi0fit_mass[:, 0]
        return self._pi0_mass

    @property
    def pi0_gamma1_momentum(self):
        """Fit momentum of first photon"""
        if self._pi0_gamma1_momentum is None:
            self._pi0_gamma1_momentum = self._pi0fit_gamma1_momentum[:, 0]
        return self._pi0_gamma1_momentum

    @property
    def pi0_gamma2_momentum(self):
        """Fit momentum of second photon"""
        if self._pi0_gamma2_momentum is None:
            self._pi0_gamma2_momentum = self._pi0fit_gamma2_momentum[:, 0]
        return self._pi0_gamma2_momentum

    @property
    def pi0_gamma1_direction(self):
        """Fit direction of the first photon"""
        if self._pi0_gamma1_direction is None:
            self._pi0_gamma1_direction = self._pi0fit_gamma1_direction[:, 0, :]
        return self._pi0_gamma1_direction

    @property
    def pi0_gamma2_direction(self):
        """Fit direction of the second photon"""
        if self._pi0_gamma2_direction is None:
            self._pi0_gamma2_direction = self._pi0fit_gamma2_direction[:, 0, :]
        return self._pi0_gamma2_direction

    @property
    def pi0_gamma1_conversion_length(self):
        """Fit conversion length for the first photon"""
        if self._pi0_gamma1_conversion_length is None:
            self._pi0_gamma1_conversion_length = self._pi0fit_gamma1_conversion_length[:, 0]
        return self._pi0_gamma1_conversion_length

    @property
    def pi0_gamma2_conversion_length(self):
        """Fit conversion length for the second photon"""
        if self._pi0_gamma2_conversion_length is None:
            self._pi0_gamma2_conversion_length = self._pi0fit_gamma2_conversion_length[:, 0]
        return self._pi0_gamma2_conversion_length

    @property
    def pi0_gamma_opening_angle(self):
        """Fit opening angle between the photons"""
        if self._pi0_gamma_opening_angle is None:
            self._pi0_gamma_opening_angle = self._pi0fit_gamma_opening_angle[:, 0]
        return self._pi0_gamma_opening_angle
