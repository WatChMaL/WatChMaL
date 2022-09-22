import uproot
import glob
import numpy as np
from abc import ABC, abstractmethod
from os.path import dirname
from omegaconf import OmegaConf
from matplotlib import pyplot as plt


class WatChMaLOutput(ABC):
    def __init__(self, directory, indices=None):
        self.directory = directory
        self.indices = indices
        self._training_log = None
        self._train_log_epoch = None
        self._train_log_loss = None
        self._val_log_epoch = None
        self._val_log_loss = None
        self._val_log_best = None

    def plot_training_progression(self, plot_best=True, y_lim=None, fig_size=None, title=None, legend='center right'):
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
        train_files = glob.glob(self.directory + "/outputs/log_train*.csv")
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
        ndarray
            Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
            contains the softmax values of a class.
        """
        outputs = np.load(self.directory + "/outputs/" + name + ".npy")
        output_indices = np.load(self.directory + "/outputs/indices.npy")
        if self.indices is None:
            return outputs[output_indices.argsort()].squeeze()
        intersection = np.intersect1d(self.indices, output_indices, return_indices=True)
        sorted_outputs = np.zeros(self.indices.shape + outputs.shape[1:])
        sorted_outputs[intersection[1]] = outputs[intersection[2]]
        return sorted_outputs.squeeze()

    @abstractmethod
    def read_training_log_from_csv(self, directory):
        """This method should load the training log, set the corresponding attributes and return a tuple of them."""

    @property
    def training_log(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._training_log

    @property
    def train_log_epoch(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_epoch

    @property
    def train_log_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_loss

    @property
    def val_log_epoch(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_epoch

    @property
    def val_log_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_loss

    @property
    def val_log_best(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_best


class FiTQunOutput:
    def __init__(self, file_path):
        self.chain = uproot.lazy(file_path)

        self.n_timewindows = self.chain['fqntwnd']                  # Number of time windows (good clusters) in this event
        self.timewindow = self.chain['fqtwnd']                      # Cluster index of the time window(corresponds to cluster_ncand)
        self.timewindow_cluster = self.chain['fqtwnd_iclstr']       # Number of peaks(sub-events) in the time window
        self.timewindow_time = self.chain['fqtwnd_prftt0']          # Pre-fitter vertex time
        self.timewindow_position = self.chain['fqtwnd_prftpos']     # Pre-fitter vertex position
        self.timewindow_npeaks = self.chain['fqtwnd_npeak']         # Time window start/end time
        self.timewindow_peak_time = self.chain['fqtwnd_peakt0']     # Time of each sub-event in the time window
        self.timewindow_peakiness = self.chain['fqtwnd_peakiness']  # Vertex goodness parameter evaluated at the peak position

        self.n_subevents = self.chain['fqnse']                      # Total number of subevents in the event
        self.subevent_timewindow = self.chain['fqitwnd']            # Index of the time window to which the subevent belongs
        self.subevent_peak = self.chain['fqipeak']                  # Peak index within the time window
        self.subevent_nhits = self.chain['fqnhitpmt']               # Number of hits in each subevent
        self.subevent_total_charge = self.chain['fqtotq']           # Total charge in each subevent
        self.subevent_0ring_total_charge = self.chain['fq0rtotmu']  # Total predicted charge for the 0-ring hypothesis in each subevent - these variables are the result of evaluating the likelihood function with a particle that is below Cherenkov threshold.
        self.subevent_0ring_nll = self.chain['fq0rnll']             # -log(L) value for the 0-ring hypothesis in each subevent
        self.subevent_n50 = self.chain['fqn50']                     # n50 - In the TOF-corrected hit time distribution, number of hits within the 50ns time window centred at vertex time(1R electron fit vertex is used)
        self.subevent_q50 = self.chain['fqq50']                     # q50 - Total charge of hits included in n50 above

        # These variables are the result of the 1-ring fits. The first index is the subevent (1-ring fits are run on all subevents). The second
        # index is the particle-hypothesis index (same as apfit):
        # 0 = GAMMA, 1 = ELECTRON, 2 = MUON, 3 = PION, 4 = KAON, 5 = PROTON,  6 = CONE GENERATOR
        # Currently, only the electron, muon, and pion (the upstream pion segment) hypotheses are implemented.
        self._singlering_flag = self.chain['fq1rpcflg']               # Flag to indicate whether fiTQun believes the particle is exiting the ID(<0 if MINUIT did not converge)
        self._singlering_momentum = self.chain['fq1rmom']             # Fit momentum
        self._singlering_position = self.chain['fq1rpos']             # Fit vertex (0=X, 1=Y, 2=Z)
        self._singlering_direction = self.chain['fq1rdir']            # Fit direction (0=X, 1=Y, 2=Z)
        self._singlering_time = self.chain['fq1rt0']                  # Fit particle creation time
        self._singlering_total_charge = self.chain['fq1rtotmu']       # Best-fit total predicted charge
        self._singlering_nll = self.chain['fq1rnll']                  # Best-fit -lnL
        self._singlering_conversion_length = self.chain['fq1rdconv']  # Fit conversion length (always 0 for 1R fits)
        self._singlering_energy_loss = self.chain['fq1reloss']        # Energy lost in the upstream track segment before the hadronic interaction(for upstream tracks only)

        self._electron_flag = None                                    # Flag to indicate whether fiTQun believes the particle is exiting the ID(<0 if MINUIT did not converge)
        self._electron_momentum = None                                # Fit momentum
        self._electron_position = None                                # Fit vertex (0=X, 1=Y, 2=Z)
        self._electron_direction = None                               # Fit direction (0=X, 1=Y, 2=Z)
        self._electron_time = None                                    # Fit particle creation time
        self._electron_total_charge = None                            # Best-fit total predicted charge
        self._electron_nll = None                                     # Best-fit -lnL
        self._electron_conversion_length = None                       # Fit conversion length (always 0 for 1R fits)
        self._electron_energy_loss = None                             # Energy lost in the upstream track segment before the hadronic interaction(for upstream tracks only)

        self._muon_flag = None                                        # Flag to indicate whether fiTQun believes the particle is exiting the ID(<0 if MINUIT did not converge)
        self._muon_momentum = None                                    # Fit momentum
        self._muon_position = None                                    # Fit vertex (0=X, 1=Y, 2=Z)
        self._muon_direction = None                                   # Fit direction (0=X, 1=Y, 2=Z)
        self._muon_time = None                                        # Fit particle creation time
        self._muon_total_charge = None                                # Best-fit total predicted charge
        self._muon_nll = None                                         # Best-fit -lnL
        self._muon_conversion_length = None                           # Fit conversion length (always 0 for 1R fits)
        self._muon_energy_loss = None                                 # Energy lost in the upstream track segment before the hadronic interaction(for upstream tracks only)

        # Pi0 fits are only run on the first subevent. Index 0 gives the standard, unconstrained pi0 fit. (Index 1 is not filled currently)
        self._pi0fit_flag = self.chain['fqpi0pcflg']                       # (PCflg for photon 1) + 2*(PCflg for photon 2)
        self._pi0fit_momentum = self.chain['fqpi0momtot']                  # Fit momentum of the pi0
        self._pi0fit_position = self.chain['fqpi0pos']                     # Fit vertex position
        self._pi0fit_direction = self.chain['fqpi0dirtot']                 # Fit direction of the pi0
        self._pi0fit_time = self.chain['fqpi0t0']                          # Fit pi0 creation time
        self._pi0fit_total_charge = self.chain['fqpi0totmu']               # Best fit total predicted charge
        self._pi0fit_nll = self.chain['fqpi0nll']                          # Best fit -log(Likelihood)
        self._pi0fit_mass = self.chain['fqpi0mass']                        # Fit pi0 mass (always 134.9766 for constrained mass fit)
        self._pi0fit_gamma1_momentum = self.chain['fqpi0mom1']             # Fit momentum of first photon
        self._pi0fit_gamma2_momentum = self.chain['fqpi0mom2']             # Fit momentum of second photon
        self._pi0fit_gamma1_direction = self.chain['fqpi0dir1']            # Fit direction of the first photon
        self._pi0fit_gamma2_direction = self.chain['fqpi0dir2']            # Fit direction of the second photon
        self._pi0fit_gamma1_conversion_length = self.chain['fqpi0dconv1']  # Fit conversion length for the first photon
        self._pi0fit_gamma2_conversion_length = self.chain['fqpi0dconv2']  # Fit conversion length for the second photon
        self._pi0fit_gamma_opening_angle = self.chain['fqpi0photangle']    # Fit opening angle between the photons

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

        # These are the results of the Multi-Ring(MR) fits. The number of executed multi-ring fits depends on the event topology,
        # and the first index specifies different fit results.(Index 0 is the best-fit result.)
        # Each fit result is assigned a unique fit ID which tells the type of the fit(see fiTQun.cc for more details):

        #   8-digit ID "N0...ZYX" :
        #     These are the raw MR fitter output, in which a ring is either an electron or a pi+.
        #     The most significant digit "N" is the number of rings(1-6), and X, Y and Z are the particle type(as in 1R fit, "1" for e, "3" for pi+)
        #     of the 1st, 2nd and 3rd ring respectively.
        #     Negative fit ID indicates that the ring which is newly added in the fit is determined as a fake ring by the fake ring reduction algorithm.

        #   9-digit ID "1N0...ZYX" :
        #     These are the results after the fake ring reduction is applied on the raw MR fit results above with ID "N0...ZYX". Rings are re-ordered according to
        #     their visible energy, and one needs refer to the fqmrpid variable for the particle type of each ring, not the fit ID.

        #   9-digit ID "2N0...ZYX" :
        #     These are the results after the fake ring merging and sequential re-fitting are applied on the post-reduction result "1N0...ZYX".
        #     PID of a ring can change after the re-fit, and muon hypothesis is also applied on the most energetic ring.

        #   9-digit ID "3N0...ZYX" :
        #     These are the results after simultaneously fitting the longitudinal vertex position and the visible energy of all rings,
        #     on the post-refit result "2N0...ZYX".(Still experimental)

        #   9-digit ID "8NX000000" :
        #     When the best-fit hypothesis has more than one ring, the negative log-likelihood values for each ring (N) and PID hypothesis (X) can be obtained using these results. For example, to compare the likelihood for the pion and electron hypotheses of the second ring, the IDs "813000000" and "811000000" could be used.
        self.n_multiring_fits = self.chain['fqnmrfit']                # Number of MR fit results that are available
        self.multiring_fit_id = self.chain['fqmrifit']                # Fit ID of each MR fit result
        self.multiring_n_rings = self.chain['fqmrnring']              # Number of rings for this fit [1-6]
        self.multiring_flag = self.chain['fqmrpcflg']                 # <0 if MINUIT did not converge during the fit
        self.multiring_pid = self.chain['fqmrpid']                    # Particle type index for each ring in this fit (Same convention as in 1R fit)
        self.multiring_momentum = self.chain['fqmrmom']               # Fit momentum of each ring
        self.multiring_position = self.chain['fqmrpos']               # Fit vertex position of each ring
        self.multiring_direction = self.chain['fqmrdir']              # Fit direction of each ring
        self.multiring_time = self.chain['fqmrt0']                    # Fit creation time of each ring
        self.multiring_total_charge = self.chain['fqmrtotmu']         # Best-fit total predicted charge
        self.multiring_nll = self.chain['fqmrnll']                    # Best-fit -lnL
        self.multiring_conversion_length = self.chain['fqmrdconv']    # Fit conversion length of each ring(always "0" in default mode)
        self.multiring_energy_loss = self.chain['fqmreloss']          # Energy lost in the upstream track segment(for upstream tracks only)

        # These are the results of the Multi-Segment(M-S) muon fits. By default, the stand-alone M-S fit(first index="0") is applied on every event,
        # and if the most energetic ring in the best-fit MR fit is a muon, the muon track is re-fitted as a M-S track.(first index="1")
        self.n_multisegment_fits = self.chain['fqmsnfit']             # Number of Multi-Segment fit results that are available
        self.multisegment_flag = self.chain['fqmspcflg']              # <0 if MINUIT did not converge during the fit
        self.multisegment_n_segments = self.chain['fqmsnseg']         # Number of track segments in the fit
        self.multisegment_pid = self.chain['fqmspid']                 # Particle type of the M-S track (always "2")
        self.multisegment_fit_id = self.chain['fqmsifit']             # Fit ID of the MR fit that seeded this fit("1" for the stand-alone M-S fit)
        self.multisegment_ring = self.chain['fqmsimer']               # Index of the ring to which the M-S track corresponds in the seeding MR fit
        self.multisegment_momentum = self.chain['fqmsmom']            # Fit initial momentum of each segment
        self.multisegment_position = self.chain['fqmspos']            # Fit vertex position of each segment
        self.multisegment_direction = self.chain['fqmsdir']           # Fit direction of each segment
        self.multisegment_time = self.chain['fqmst0']                 # Fit creation time of each segment
        self.multisegment_total_charge = self.chain['fqmstotmu']      # Best-fit total predicted charge
        self.multisegment_nll = self.chain['fqmsnll']                 # Best-fit -lnL
        self.multisegment_energy_loss = self.chain['fqmseloss']       # Energy lost in each segment

        # These are the results of the PDK_MuGamma fit, dedicated to proton decay
        # searches. Although there are two available fit results for each quantity, only
        # the first is used (e.g. fqpmgmom1[0]).
        self.pdk_flag = self.chain['fqpmgpcflg']                      # (PCflg for muon) + 2*(PCflg for gamma)
        self.pdk_muon_momentum = self.chain['fqpmgmom1']              # Best-fit muon momentum
        self.pdk_muon_position = self.chain['fqpmgpos1']              # Best-fit muon position
        self.pdk_muon_direction = self.chain['fqpmgdir1']             # Best-fit muon direction
        self.pdk_muon_time = self.chain['fqpmgt01']                   # Best-fit muon time
        self.pdk_gamma_momentum = self.chain['fqpmgmom2']             # Best-fit gamma momentum
        self.pdk_gamma_position = self.chain['fqpmgpos2']             # Best-fit gamma position
        self.pdk_gamma_direction = self.chain['fqpmgdir2']            # Best-fit gamma direction
        self.pdk_gamma_time = self.chain['fqpmgt02']                  # Best-fit gamma time
        self.pdk_total_charge = self.chain['fqpmgtotmu']              # Best-fit total predicted charge
        self.pdk_nll = self.chain['fqpmgnll']                         # Best-fit negative log-likelihood

    @property
    def electron_flag(self):
        if self._electron_flag is None:
            self._electron_flag = self._singlering_flag[:, 0, 1]
        return self._electron_flag

    @property
    def electron_momentum(self):
        if self._electron_momentum is None:
            self._electron_momentum = self._singlering_momentum[:, 0, 1]
        return self._electron_momentum

    @property
    def electron_position(self):
        if self._electron_position is None:
            self._electron_position = self._singlering_position[:, 0, 1, :]
        return self._electron_position

    @property
    def electron_direction(self):
        if self._electron_direction is None:
            self._electron_direction = self._singlering_direction[:, 0, 1, :]
        return self._electron_direction

    @property
    def electron_time(self):
        if self._electron_time is None:
            self._electron_time = self._singlering_time[:, 0, 1]
        return self._electron_time

    @property
    def electron_total_charge(self):
        if self._electron_total_charge is None:
            self._electron_total_charge = self._singlering_total_charge[:, 0, 1]
        return self._electron_total_charge

    @property
    def electron_nll(self):
        if self._electron_nll is None:
            self._electron_nll = self._singlering_nll[:, 0, 1]
        return self._electron_nll

    @property
    def electron_conversion_length(self):
        if self._electron_conversion_length is None:
            self._electron_conversion_length = self._singlering_conversion_length[:, 0, 1]
        return self._electron_conversion_length

    @property
    def electron_energy_loss(self):
        if self._electron_energy_loss is None:
            self._electron_energy_loss = self._singlering_energy_loss[:, 0, 1]
        return self._electron_energy_loss

    @property
    def muon_flag(self):
        if self._muon_flag is None:
            self._muon_flag = self._singlering_flag[:, 0, 2]
        return self._muon_flag

    @property
    def muon_momentum(self):
        if self._muon_momentum is None:
            self._muon_momentum = self._singlering_momentum[:, 0, 2]
        return self._muon_momentum

    @property
    def muon_position(self):
        if self._muon_position is None:
            self._muon_position = self._singlering_position[:, 0, 2, :]
        return self._muon_position

    @property
    def muon_direction(self):
        if self._muon_direction is None:
            self._muon_direction = self._singlering_direction[:, 0, 2, :]
        return self._muon_direction

    @property
    def muon_time(self):
        if self._muon_time is None:
            self._muon_time = self._singlering_time[:, 0, 2]
        return self._muon_time

    @property
    def muon_total_charge(self):
        if self._muon_total_charge is None:
            self._muon_total_charge = self._singlering_total_charge[:, 0, 2]
        return self._muon_total_charge

    @property
    def muon_nll(self):
        if self._muon_nll is None:
            self._muon_nll = self._singlering_nll[:, 0, 2]
        return self._muon_nll

    @property
    def muon_conversion_length(self):
        if self._muon_conversion_length is None:
            self._muon_conversion_length = self._singlering_conversion_length[:, 0, 2]
        return self._muon_conversion_length

    @property
    def muon_energy_loss(self):
        if self._muon_energy_loss is None:
            self._muon_energy_loss = self._singlering_energy_loss[:, 0, 2]
        return self._muon_energy_loss

    @property
    def pi0_flag(self):
        if self._pi0_flag is None:
            self._pi0_flag = self._pi0fit_flag[:, 0]
        return self._pi0_flag

    @property
    def pi0_momentum(self):
        if self._pi0_momentum is None:
            self._pi0_momentum = self._pi0fit_momentum[:, 0]
        return self._pi0_momentum

    @property
    def pi0_position(self):
        if self._pi0_position is None:
            self._pi0_position = self._pi0fit_position[:, 0, :]
        return self._pi0_position

    @property
    def pi0_direction(self):
        if self._pi0_direction is None:
            self._pi0_direction = self._pi0fit_direction[:, 0, :]
        return self._pi0_direction

    @property
    def pi0_time(self):
        if self._pi0_time is None:
            self._pi0_time = self._pi0fit_time[:, 0]
        return self._pi0_time

    @property
    def pi0_total_charge(self):
        if self._pi0_total_charge is None:
            self._pi0_total_charge = self._pi0fit_total_charge[:, 0]
        return self._pi0_total_charge

    @property
    def pi0_nll(self):
        if self._pi0_nll is None:
            self._pi0_nll = self._pi0fit_nll[:, 0]
        return self._pi0_nll

    @property
    def pi0_mass(self):
        if self._pi0_mass is None:
            self._pi0_mass = self._pi0fit_mass[:, 0]
        return self._pi0_mass

    @property
    def pi0_gamma1_momentum(self):
        if self._pi0_gamma1_momentum is None:
            self._pi0_gamma1_momentum = self._pi0fit_gamma1_momentum[:, 0]
        return self._pi0_gamma1_momentum

    @property
    def pi0_gamma2_momentum(self):
        if self._pi0_gamma2_momentum is None:
            self._pi0_gamma2_momentum = self._pi0fit_gamma2_momentum[:, 0]
        return self._pi0_gamma2_momentum

    @property
    def pi0_gamma1_direction(self):
        if self._pi0_gamma1_direction is None:
            self._pi0_gamma1_direction = self._pi0fit_gamma1_direction[:, 0, :]
        return self._pi0_gamma1_direction

    @property
    def pi0_gamma2_direction(self):
        if self._pi0_gamma2_direction is None:
            self._pi0_gamma2_direction = self._pi0fit_gamma2_direction[:, 0, :]
        return self._pi0_gamma2_direction

    @property
    def pi0_gamma1_conversion_length(self):
        if self._pi0_gamma1_conversion_length is None:
            self._pi0_gamma1_conversion_length = self._pi0fit_gamma1_conversion_length[:, 0]
        return self._pi0_gamma1_conversion_length

    @property
    def pi0_gamma2_conversion_length(self):
        if self._pi0_gamma2_conversion_length is None:
            self._pi0_gamma2_conversion_length = self._pi0fit_gamma2_conversion_length[:, 0]
        return self._pi0_gamma2_conversion_length

    @property
    def pi0_gamma_opening_angle(self):
        if self._pi0_gamma_opening_angle is None:
            self._pi0_gamma_opening_angle = self._pi0fit_gamma_opening_angle[:, 0]
        return self._pi0_gamma_opening_angle