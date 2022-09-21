import numpy as np
import glob
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from os.path import dirname
from sklearn import metrics
from omegaconf import OmegaConf

import analysis.utils.binning as bins
import analysis.utils.plotting as plot
from analysis.read import WatChMaLOutput


def combine_softmax(softmaxes, labels):
    """
    Sum the softmax values for the given labels.

    Parameters
    ----------
    softmaxes: ndarray
        Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
        contains the softmax values of a class.
    labels: int or sequence of ints
        Set of labels (corresponding to classes) to combine. Can be just a single label, in which case the corresponding
        column of `softmaxes` is returned.

    Returns
    -------
    ndarray
        One-dimensional array of summed softmax values, with length equal to the first dimension of `softmaxes`.
    """
    labels = np.atleast_1d(labels)
    return np.sum(softmaxes[:, labels], axis=1)


def plot_rocs(runs, signal, signal_labels, background_labels, selection=..., fig_size=None, x_label="", y_label="",
              x_lim=None, y_lim=None, y_log=None, x_log=None, legend='best', mode='rejection', **plot_args):
    """
    Plot overlaid ROC curves of results from a number of classification runs

    Parameters
    ----------
    runs: sequence of ClassificationRun
        Sequence of runs to plot
    signal: array_like of bool
        One dimensional array of boolean values, the same length as the number of events in each run, indicating whether
        each event is classed as signal.
    signal_labels: int or sequence of ints
        Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
    background_labels: int or sequence of ints
        Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.
    selection: indexing expression, optional
        Selection of the discriminator values to be used (by default use all values).
    fig_size: (float, float), optional
        Figure size.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    x_lim: (float, float), optional
        Limits of the x-axis.
    y_lim: (float, float), optional
        Limits of the y-axis.
    x_log: bool, optional
        If True, plot the x-axis with log scale, otherwise use linear scale (default).
    y_log: str, optional
        If True, plot the y-axis with log scale (default for 'rejection' mode, otherwise use linear scale (default for
        'efficiency' mode.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    mode: {'rejection', 'efficiency'}, optional
        If `rejection` (default) plot rejection factor (reciprocal of the false positive rate) on the y-axis versus
        signal efficiency (true positive rate) on the x-axis. If `efficiency` plot background mis-ID rate (false
        positive rate) versus signal efficiency (true positive rate) on the x-axis.
    plot_args: optional
        Additional arguments to pass to the `hist` plotting function. Note that these may be overridden by arguments
        defined in `runs`.

    Returns
    -------
    fig: Figure
    ax: axes.Axes
    """
    fig, ax = plt.subplots(figsize=fig_size)
    selected_signal = signal[selection]
    for r in runs:
        selected_discriminator = r.discriminator(signal_labels, background_labels)[selection]
        fpr, tpr, _ = metrics.roc_curve(selected_signal, selected_discriminator)
        auc = metrics.auc(fpr, tpr)
        args = {**plot_args, **r.plot_args}
        args['label'] = f"{args['label']} (AUC={auc:.4f})"
        if mode == 'rejection':
            if y_log is None:
                y_log = True
            with np.errstate(divide='ignore'):
                ax.plot(tpr, 1/fpr, **args)
        elif mode == 'efficiency':
            ax.plot(fpr, tpr, **args)
        else:
            raise ValueError(f"Unknown ROC curve mode '{mode}'.")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    if y_lim:
        ax.set_ylim(y_lim)
    if x_lim:
        ax.set_xlim(x_lim)
    if legend:
        ax.legend(loc=legend)
    return fig, ax


def plot_efficiency_profile(runs, binning, selection=..., fig_size=None, x_label="", y_label="", legend='best',
                            y_lim=None, **plot_args):
    """
    Plot binned efficiencies for a cut applied to a number of classification runs.
    Each run should already have had a cut generated, then in each bin the proportion of events passing the cut is
    calculated as the efficiency and plotted. A selection can be provided to use only a subset of all the values. The
    same binning and selection is applied to each run.

    Parameters
    ----------
    runs: sequence of ClassificationRun
        Sequence of runs to plot
    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    fig_size: (float, float), optional
        Figure size.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    y_lim: (float, float), optional
        Limits of the y-axis.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: Figure
    ax: axes.Axes
    """
    fig, ax = plt.subplots(figsize=fig_size)
    for r in runs:
        args = {**plot_args, **r.plot_args}
        r.plot_binned_efficiency(ax, binning, selection, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


class ClassificationRun(ABC):
    def __init__(self, run_label, plot_args=None):
        self.label = run_label
        if plot_args is None:
            plot_args = {}
        plot_args['label'] = run_label
        self.plot_args = plot_args
        self.cut = None

    @abstractmethod
    def discriminator(self, signal_labels, background_labels):
        """This method should return the discriminator for the given signal and background labels"""

    def cut_with_constant_binned_efficiency(self, signal_labels, background_labels, efficiency, binning, selection=...,
                                            return_thresholds=False):
        """
        Generate array of boolean values indicating whether each event passes a cut defined such that, in each bin of
        some binning of the events, a constant proportion of the selected events pass the cut.
        After taking the subset of `discriminator_values` defined by `selection`, in each bin of `binning` the threshold
        discriminator value is found such that the proportion that are above the threshold is equal to `efficiency`.
        These cut thresholds are then used to apply the cut to all events (not just those selected by `selection`) and
        an array of booleans is returned for whether each discriminator value is above the threshold of its
        corresponding bin. The cut result is also stored for use in plotting efficiency profiles.

        Parameters
        ----------
        signal_labels: int or sequence of ints
            Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
        background_labels: int or sequence of ints
            Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.
        efficiency: float
            The fixed efficiency to ensure in each bin.
        binning: (ndarray, ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the discriminator values to use in calculating the thresholds applied by the cut in each bin
            (by default use all values).
        return_thresholds:
            If True, return also the array of cut thresholds calculated for each bin.

        Returns
        -------
        cut: ndarray of bool
            One-dimensional array, length of the total number of events, indicating whether each event passes the cut.
        thresholds: ndarray of float, optional
            One-dimensional array giving the threshold applied by the cut to events in each bin.
        """
        discriminator_values = self.discriminator(signal_labels, background_labels)
        binned_discriminators = bins.apply_binning(discriminator_values, binning, selection)
        thresholds = bins.binned_quantiles(binned_discriminators, 1 - efficiency)
        # put inf as first and last threshold for overflow bins
        padded_thresholds = np.concatenate(([np.inf], thresholds, [np.inf]))
        self.cut = np.array(discriminator_values) > padded_thresholds[binning[1]]
        if return_thresholds:
            return self.cut, thresholds
        else:
            return self.cut

    def cut_with_fixed_efficiency(self, signal_labels, background_labels, efficiency, selection=...,
                                  return_threshold=False):
        """
        Generate array of boolean values indicating whether each event passes a cut defined such that a fixed proportion
        of the selected events pass the cut.
        After taking the subset of `discriminator_values` defined by `selection`, the threshold discriminator value is
        found such that the proportion that are above the threshold is equal to `efficiency`. This cut threshold is then
        used to apply the cut to all events (not just those selected by `selection`) and an array of booleans is
        returned for whether each discriminator value is above the threshold of its corresponding bin. The cut result is
        also stored for use in plotting efficiency profiles.

        Parameters
        ----------
        signal_labels: int or sequence of ints
            Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
        background_labels: int or sequence of ints
            Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.
        efficiency: float
            The fixed efficiency.
        selection: indexing expression, optional
            Selection of the discriminator values to use in calculating the threshold applied by the cut (by default use
            all values).
        return_threshold: bool, optional
            If True, return also the cut threshold.

        Returns
        -------
        cut: ndarray of bool
            One-dimensional array the same length as `discriminator_values` indicating whether each event passes the cut.
        threshold: float, optional
            The threshold applied by the cut.
        """
        discriminator_values = self.discriminator(signal_labels, background_labels)
        threshold = np.quantile(discriminator_values[selection], 1 - efficiency)
        self.cut = np.array(discriminator_values) > threshold
        if return_threshold:
            return self.cut, threshold
        else:
            return self.cut

    def plot_binned_efficiency(self, ax, binning, selection=..., reverse=False, errors=False, x_errors=True,
                               **plot_args):
        """
        Plot binned efficiencies of the cut applied to the classification run on an existing set of axes.
        The cut values corresponding to booleans indicating whether each event passes the cut are divided up into bins
        of some quantity according to `binning`, before calculating the efficiency (proportion of events passing the
        cut) in each bin. A selection can be provided to use only a subset of all the values.

        Parameters
        ----------
        ax: axes.Axes
            Axes to draw the plot.
        binning: (ndarray, ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the values to use in calculating the resolutions (by default use all values).
        reverse: bool
            If True, reverse the cut to plot percentage of events failing the cut. By default the percentage of events
            passing the cut is plotted
        errors: bool, optional
            If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
        x_errors: bool, optional
            If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
        plot_args: optional
            Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
            provided in `runs`.
        """
        plot_args.setdefault('lw', 2)
        binned_cut = bins.apply_binning(self.cut, binning, selection)
        y = bins.binned_efficiencies(binned_cut, errors, reverse=reverse)
        x = bins.bin_centres(binning[0])
        if errors:
            y_values, y_errors = y
            x_errors = bins.bin_halfwidths(binning[0]) if x_errors else None
            plot_args.setdefault('marker', '')
            plot_args.setdefault('capsize', 4)
            plot_args.setdefault('capthick', 2)
            ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, **plot_args)
        else:
            plot_args.setdefault('marker', 'o')
            ax.plot(x, y, **plot_args)


class WatChMaLClassification(ClassificationRun, WatChMaLOutput):
    def __init__(self, directory, run_label, indices=None, **plot_args):
        """
        Constructs the object holding the results of a WatChMaL classification run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL classification run.
        run_label: str
            Label this run to use in plot legends, etc.
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        plot_args: optional
            Additional arguments to pass to plotting functions.
        """
        ClassificationRun.__init__(self, run_label=run_label, plot_args=plot_args)
        WatChMaLOutput.__init__(self, directory)
        self.indices = indices
        self._softmaxes = None
        self._train_log_accuracy = None
        self._val_log_accuracy = None

    def read_training_log(self):
        train_files = glob.glob(self.directory + "/outputs/log_train*.csv")
        if train_files:
            return self.read_training_log_from_csv(self.directory)
        else:  # search for a previous training run with a saved state that was loaded
            conf = OmegaConf.load(self.directory + '/.hydra/config.yaml')
            state_file = conf.tasks.restore_state.weight_file
            directory = dirname(dirname(state_file))
            return self.read_training_log_from_csv(directory)

    def read_training_log_from_csv(self, directory):
        train_files = glob.glob(self.directory + "/outputs/log_train*.csv")
        log_train = np.array([np.genfromtxt(f, delimiter=',', skip_header=1) for f in train_files])
        log_val = np.genfromtxt(directory + "/outputs/log_val.csv", delimiter=',', skip_header=1)
        train_iteration = log_train[0, :, 0]
        train_epoch = log_train[0, :, 1]
        it_per_epoch = np.min(train_iteration[train_epoch == 1]) - 1
        self._train_log_epoch = train_iteration / it_per_epoch
        self._train_log_loss = np.mean(log_train[:, :, 2], axis=0)
        self._train_log_accuracy = np.mean(log_train[:, :, 3], axis=0)
        self._val_log_epoch = log_val[:, 0] / it_per_epoch
        self._val_log_loss = log_val[:, 1]
        self._val_log_accuracy = log_val[:, 2]
        self._val_log_best = log_val[:, 3].astype(bool)
        return (self._train_log_epoch, self._train_log_loss, self._train_log_accuracy,
                self._val_log_epoch, self._val_log_loss, self._val_log_accuracy, self._val_log_best)

    def get_softmaxes(self):
        """
        Read the softmax predictions resulting from the evaluation run of a WatChMaL classification model.

        Returns
        -------
        ndarray
            Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
            contains the softmax values of a class.
        """
        softmaxes = np.load(self.directory + "/outputs/softmax.npy")
        output_indices = np.load(self.directory + "/outputs/indices.npy")
        if self.indices is None:
            return softmaxes[output_indices.argsort()].squeeze()
        intersection = np.intersect1d(self.indices, output_indices, return_indices=True)
        sorted_softmaxes = np.zeros(self.indices.shape + softmaxes.shape[1:])
        sorted_softmaxes[intersection[1]] = softmaxes[intersection[2]]
        return sorted_softmaxes.squeeze()

    def plot_training_progression(self, plot_best=True, y_loss_lim=None, fig_size=None, title=None,
                                  legend='center right'):
        fig, ax1 = super().plot_training_progression(plot_best, y_loss_lim, fig_size, title, legend)
        ax2 = ax1.twinx()
        ax2.plot(self.train_log_epoch, self.train_log_accuracy, lw=2, label='Train accuracy', color='r', alpha=0.3)
        ax2.plot(self.val_log_epoch, self.val_log_accuracy, lw=2, label='Validation accuracy', color='r')
        if plot_best:
            ax2.plot(self.val_log_epoch[self.val_log_best], self.val_log_accuracy[self.val_log_best], lw=0, marker='o',
                     label='Best validation accuracy', color='darkred')
        ax2.set_ylabel("Accuracy", c='r')
        if legend:
            ax1.legend(plot.combine_legends((ax1, ax2)), loc=legend)
        return fig, ax1, ax2

    def discriminator(self, signal_labels, background_labels):
        """
        Return a discriminator with appropriate scaling of softmax values from multi-class training, given the set of
        signal and background class labels. For each event, the discriminator is the sum the signal softmax values
        normalised by the sum of signal and background softmax values.

        Parameters
        ----------
        signal_labels: int or sequence of ints
            Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
        background_labels: int or sequence of ints
            Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.

        Returns
        -------
        ndarray
            One-dimensional array of discriminator values, with length equal to the first dimension of `softmaxes`.
        """
        signal_softmax = combine_softmax(self.softmaxes, signal_labels)
        background_softmax = combine_softmax(self.softmaxes, background_labels)
        return signal_softmax / (signal_softmax + background_softmax)

    @property
    def train_log_accuracy(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_accuracy

    @property
    def val_log_accuracy(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_accuracy

    @property
    def softmaxes(self):
        if self._softmaxes is None:
            self._softmaxes = self.get_softmaxes()
        return self._softmaxes


class FiTQunClassification(ClassificationRun):

    def __init__(self, fitqun_output, run_label, indices=None, particle_labels=None, **plot_args):
        super().__init__(run_label=run_label, plot_args=plot_args)
        self.fitqun_output = fitqun_output
        self.indices = indices
        if particle_labels is None:
            particle_labels = {'gamma': 0, 'electron': 1, 'muon': 2, 'pi0': 3}
        self.particle_labels = particle_labels
        self.gammas = {particle_labels['gamma']}
        self.electrons = {particle_labels['electron']}
        self.muons = {particle_labels['muon']}
        self.pi0s = {particle_labels['pi0']}
        self.electron_like = {self.particle_labels[p] for p in ['electron', 'gamma']}
        self._electron_gamma_discriminator = None
        self._electron_muon_discriminator = None

    def discriminator(self, signal_labels, background_labels):
        if set(signal_labels) <= self.electron_like and set(background_labels) <= self.muons:
            return self.electron_muon_discriminator
        elif set(signal_labels) <= self.muons and set(background_labels) <= self.electron_like:
            return self.muon_electron_discriminator
        elif set(signal_labels) <= self.electrons and set(background_labels) <= self.gammas:
            return self.electron_gamma_discriminator
        elif set(signal_labels) <= self.gammas and set(background_labels) <= self.electrons:
            return self.gamma_electron_discriminator
        else:
            raise NotImplementedError("A discriminator for these signal and background labels has not been implemented in fiTQun")

    def set_electron_gamma_discriminator(self, discriminator):
        if callable(discriminator):
            self._electron_gamma_discriminator = discriminator(self.fitqun_output)
        else:
            self._electron_gamma_discriminator = discriminator

    @property
    def electron_muon_discriminator(self):
        if self._electron_muon_discriminator is None:
            self._electron_muon_discriminator = self.fitqun_output.muon_nll[self.indices] - self.fitqun_output.electron_nll[self.indices]
        return self._electron_muon_discriminator

    @property
    def muon_electron_discriminator(self):
        return -self.electron_muon_discriminator

    @property
    def electron_gamma_discriminator(self):
        if self._electron_gamma_discriminator is None:
            #  fiTQun gamma hypotheses doesn't work well, so just use e/mu nll by default
            self._electron_gamma_discriminator = self.muon_electron_discriminator
        return self._electron_gamma_discriminator

    @electron_gamma_discriminator.setter
    def electron_gamma_discriminator(self, discriminator):
        if callable(discriminator):
            self._electron_gamma_discriminator = discriminator(self.fitqun_output)
        else:
            self._electron_gamma_discriminator = discriminator

    @property
    def gamma_electron_discriminator(self):
        return -self.electron_gamma_discriminator
