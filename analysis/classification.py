import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod
from sklearn import metrics
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from analysis.utils.binning import apply_binning, binned_quantiles, binned_efficiencies
import analysis.utils.plotting as plot
from analysis.read import WatChMaLOutput


def combine_softmax(softmaxes, labels, label_map=None):
    """
    Sum the softmax values for the given labels.

    Parameters
    ----------
    softmaxes: np.ndarray
        Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
        contains the softmax values of a class.
    labels: int or sequence of ints
        Set of labels (corresponding to classes) to combine. Can be just a single label, in which case the corresponding
        column of `softmaxes` is returned.
    label_map: dictionary
        Mapping from labels to columns of the softmax array. By default, assume labels map directly to column indices.

    Returns
    -------
    np.ndarray
        One-dimensional array of summed softmax values, with length equal to the first dimension of `softmaxes`.
    """
    labels = np.atleast_1d(labels)
    if label_map is not None:
        labels = [label_map[l] for l in labels]
    return np.sum(softmaxes[:, labels], axis=1)


def plot_rocs(runs, signal_labels, background_labels, selection=None, ax=None, fig_size=None, x_label="", y_label="",
              x_lim=None, y_lim=None, y_log=None, x_log=None, legend='best', mode='rejection', **plot_args):
    """
    Plot overlaid ROC curves of results from a number of classification runs

    Parameters
    ----------
    runs: sequence of ClassificationRun
        Sequence of runs to plot
    signal_labels: int or sequence of ints
        Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
    background_labels: int or sequence of ints
        Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.
    selection: indexing expression, optional
        Selection of the discriminator values to be used (by default use each run's predefined selection, or all events
        if none is defined).
    ax: matplotlib.axes.Axes
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
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
        If True, plot the y-axis with log scale (default for 'rejection' mode), otherwise use linear scale (default for
        'efficiency' mode).
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
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    for r in runs:
        run_selection = r.selection if selection is None else selection
        selected_signal = np.isin(r.true_labels, signal_labels)[run_selection]
        selected_discriminator = r.discriminator(signal_labels, background_labels)[run_selection]
        fpr, tpr, _ = metrics.roc_curve(selected_signal, selected_discriminator)
        auc = metrics.auc(fpr, tpr)
        args = {**plot_args, **r.plot_args}
        args['label'] = f"{args['label']} (AUC={auc:.4f})"
        if mode == 'rejection':
            if y_log is None:
                y_log = True
            with np.errstate(divide='ignore'):
                ax.plot(tpr, 1 / fpr, **args)
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


def plot_efficiency_profile(runs, binning, selection=None, select_labels=None, ax=None, fig_size=None, x_label="",
                            y_label="", legend='upper left', y_lim=None, label=None, **plot_args):
    """
    Plot binned efficiencies for a cut applied to a number of classification runs.
    Each run should already have had a cut generated, then in each bin the proportion of events passing the cut is
    calculated as the efficiency and plotted. A selection can be provided to use only a subset of all the values. The
    same binning and selection is applied to each run.

    Parameters
    ----------
    runs: sequence of ClassificationRun
        Sequence of runs to plot
    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the efficiencies (by default use each run's predefined selection,
        or all events if none is defined).
    select_labels: set of int, optional
        Set of true labels to select events to use.
    ax: matplotlib.axes.Axes
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
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
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    for r in runs:
        args = {**plot_args, **r.plot_args}
        run_selection = r.selection if selection is None else selection
        r.plot_binned_efficiency(ax, binning, run_selection, select_labels, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if label is not None:
        # place a text box in upper right in axes coords
        props = dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.5)
        ax.text(0.7, 0.95, label, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


class ClassificationRun(ABC):
    """Base class for classification results"""
    def __init__(self, run_label, true_labels=None, selection=None, **plot_args):
        """
        Create object to hold classification results

        Parameters
        ----------
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_labels: array_like of int, optional
            Array of true labels for the events in these classification results
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        self.run_label = run_label
        self.true_labels = true_labels
        if selection is None:
            selection = ...
        self.selection = selection
        plot_args['label'] = run_label
        self.plot_args = plot_args
        self.cut = None

    @abstractmethod
    def discriminator(self, signal_labels, background_labels):
        """This method should return the discriminator for the given signal and background labels"""

    def select_labels(self, select_labels, selection=None):
        """
        Combine a selection of events with the additional requirement of having chosen true labels.

        Parameters
        ----------
        select_labels: set of int
            Set of true labels to select
        selection: index_expression, optional
            Selection over all events (by default use the run's predefined selection)

        Returns
        -------
        np.ndarray
            Array of indices that are both selected by `selection` and have true label in `select_labels`
        """
        if selection is None:
            selection = self.selection
        if select_labels is not None:
            s = np.zeros_like(self.true_labels, dtype=bool)
            s[selection] = True
            selection = s & np.isin(self.true_labels, np.atleast_1d(select_labels))
        return selection

    def cut_with_constant_binned_efficiency(self, signal_labels, background_labels, efficiency, binning, selection=None,
                                            select_labels=None, return_thresholds=False):
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
        binning: (np.ndarray, np.ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the discriminator values to use in calculating the thresholds applied by the cut in each bin
            (by default use the run's predefined selection, or all events if none is defined).
        select_labels: set of int, optional
            Set of true labels to select events to use in calculating the thresholds.
        return_thresholds: bool, optional
            If True, return also the array of cut thresholds calculated for each bin.

        Returns
        -------
        cut: np.ndarray of bool
            One-dimensional array, length of the total number of events, indicating whether each event passes the cut.
        thresholds: np.ndarray of float, optional
            One-dimensional array giving the threshold applied by the cut to events in each bin.
        """
        selection = self.select_labels(select_labels, selection)
        discriminator_values = self.discriminator(signal_labels, background_labels)
        binned_discriminators = apply_binning(discriminator_values, binning, selection)
        thresholds = binned_quantiles(binned_discriminators, 1 - efficiency)
        # put inf as first and last threshold for overflow bins
        padded_thresholds = np.concatenate(([np.inf], thresholds, [np.inf]))
        self.cut = np.array(discriminator_values) > padded_thresholds[binning[1]]
        if return_thresholds:
            return self.cut, thresholds
        else:
            return self.cut

    def cut_with_fixed_efficiency(self, signal_labels, background_labels, efficiency, selection=None,
                                  select_labels=None, return_threshold=False):
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
        select_labels: set of int
            Set of true labels to select events to use.
        return_threshold: bool, optional
            If True, return also the cut threshold.

        Returns
        -------
        cut: np.ndarray of bool
            One-dimensional array the same length as `discriminator_values` indicating whether each event passes the cut.
        threshold: float, optional
            The threshold applied by the cut.
        """
        selection = self.select_labels(select_labels, selection)
        discriminator_values = self.discriminator(signal_labels, background_labels)
        try:
            threshold = np.quantile(discriminator_values[selection], 1 - efficiency)
        except IndexError as ex:
            raise ValueError("There are zero selected events so cannot calculate a cut with any efficiency.") from ex
        self.cut = np.array(discriminator_values) > threshold
        if return_threshold:
            return self.cut, threshold
        else:
            return self.cut

    def plot_binned_efficiency(self, ax, binning, selection=None, select_labels=None, reverse=False, errors=False,
                               x_errors=True, **plot_args):
        """
        Plot binned efficiencies of the cut applied to the classification run on an existing set of axes.
        The cut values corresponding to booleans indicating whether each event passes the cut are divided up into bins
        of some quantity according to `binning`, before calculating the efficiency (proportion of events passing the
        cut) in each bin. A selection can be provided to use only a subset of all the values.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Axes to draw the plot.
        binning: (np.ndarray, np.ndarray)
            Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
        selection: indexing expression, optional
            Selection of the values to use in calculating the resolutions (by default use all values).
        select_labels: set of int
            Set of true labels to select events to use.
        reverse: bool
            If True, reverse the cut to plot percentage of events failing the cut. By default, the percentage of events
            passing the cut is plotted
        errors: bool, optional
            If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
        x_errors: bool, optional
            If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
        plot_args: optional
            Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
            provided in `runs`.
        """
        if self.cut is None:
            raise TypeError("Cannot plot an efficiency profile for the run that has not had any cut applied. A "
                            "classification cut needs to be defined before the efficiency of that cut can be plotted.")
        
        selection = self.select_labels(select_labels, selection)

        def func(binned_cut, return_errors):
            return binned_efficiencies(binned_cut, return_errors, reverse=reverse)

        return plot.plot_binned_values(ax, func, self.cut, binning, selection, errors, x_errors, **plot_args)


class WatChMaLClassification(ClassificationRun, WatChMaLOutput):
    """Class to hold results of a WatChMaL classification run"""
    def __init__(self, directory, run_label, true_labels=None, indices=None, selection=None, **plot_args):
        """
        Constructs the object holding the results of a WatChMaL classification run.

        Parameters
        ----------
        directory: str
            Top-level output directory of a WatChMaL classification run.
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_labels: array_like of int, optional
            Array of true labels for the events in these classification results
        indices: array_like of int, optional
            Array of indices of events to select out of the indices output by WatChMaL (by default use all events sorted
            by their indices).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            By default, use all results.
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        ClassificationRun.__init__(self, run_label=run_label, true_labels=true_labels, selection=selection, **plot_args)
        WatChMaLOutput.__init__(self, directory=directory, indices=indices)
        self._softmaxes = None
        self._train_log_accuracy = None
        self._val_log_accuracy = None
        try:
            conf = OmegaConf.load(self.directory + '/.hydra/config.yaml')
            self.label_map = {l: i for i, l in enumerate(set(conf.engine.label_set))}
        except OmegaConfBaseException:
            self.label_map = None

    def read_training_log_from_csv(self, directory):
        """
        Read the training progression logs from the given directory.

        Parameters
        ----------
        directory: str
            Path to the directory of the training run.

        Returns
        -------
        np.ndarray
            Array of train epoch values for each entry in the training progression log.
        np.ndarray
            Array of train loss values for each entry in the training progression log.
        np.ndarray
            Array of train accuracy values for each entry in the training progression log.
        np.ndarray
            Array of validation epoch values for each entry in the training progression log
        np.ndarray
            Array of validation loss values for each entry in the training progression log
        np.ndarray
            Array of validation accuracy values for each entry in the training progression log
        np.ndarray
            Array of boolean values indicating whether each entry had the best validation loss so far in the training
            progression log
        """
        super().read_training_log_from_csv(directory)
        self._train_log_accuracy = np.mean(self._log_train[:, :, 3], axis=0)
        self._val_log_loss = self._log_val[:, 1]
        self._val_log_accuracy = self._log_val[:, 2]
        self._val_log_best = self._log_val[:, 3].astype(bool)
        return (self._train_log_epoch, self._train_log_loss, self._train_log_accuracy,
                self._val_log_epoch, self._val_log_loss, self._val_log_accuracy, self._val_log_best)

    def plot_training_progression(self, plot_best=True, y_loss_lim=None, fig_size=None, title=None,
                                  legend='center right', doAccuracy=True, label=None):
        """
        Plot the progression of training and validation loss and accuracy from the run's logs

        Parameters
        ----------
        plot_best: bool, optional
            If true (default), plot points indicating the best validation loss and accuracy
        y_loss_lim: (int, int), optional
            Range for the loss y-axis. By default, the range will expand to show all loss values in the logs.
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
        fig, ax1 = super().plot_training_progression(plot_best, y_loss_lim, fig_size, title, legend=None)
        # place a text box in upper right in axes coords
        props = dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.5)
        ax1.text(0.7, 0.95, label, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
        if doAccuracy:
            ax2 = ax1.twinx()
            ax2.plot(self.train_log_epoch, self.train_log_accuracy, lw=2, label='Train accuracy', color='r', alpha=0.3)
            ax2.plot(self.val_log_epoch, self.val_log_accuracy, lw=2, label='Validation accuracy', color='r')
            if plot_best:
                ax2.plot(self.val_log_epoch[self.val_log_best], self.val_log_accuracy[self.val_log_best], lw=0, marker='o',
                        label='Best validation accuracy', color='darkred')
            ax2.set_ylabel("Accuracy", c='r')
            if legend:
                ax1.legend(*plot.combine_legends((ax1, ax2)), loc=legend)
            return fig, ax1, ax2
        elif legend:
            ax1.legend( loc=legend)
        return fig, ax1

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
        np.ndarray
            One-dimensional array of discriminator values, with length equal to the number of events in this run.
        """
        signal_softmax = combine_softmax(self.softmaxes, signal_labels, self.label_map)
        background_softmax = combine_softmax(self.softmaxes, background_labels, self.label_map)
        return signal_softmax / (signal_softmax + background_softmax)

    @property
    def train_log_accuracy(self):
        """Array of train accuracy values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_accuracy

    @property
    def val_log_accuracy(self):
        """Array of validation accuracy values for each entry in the training progression log."""
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_accuracy

    @property
    def softmaxes(self):
        """Array of softmax outputs"""
        if self._softmaxes is None:
            self._softmaxes = self.get_outputs("softmax")
        return self._softmaxes


class FiTQunClassification(ClassificationRun):
    """Class to hold classification results of a fiTQun reconstruction run"""
    def __init__(self, fitqun_output, run_label, true_labels=None, indices=None, selection=None,
                 particle_label_map=None, **plot_args):
        """
        Create object containing classification results from a fiTQun reconstruction run

        Parameters
        ----------
        fitqun_output: analysis.read.FiTQunOutput
            Output from a fiTQun reconstruction run
        run_label: str
            Label to describe this set of results to use in plot legends, etc.
        true_labels: array_like of int, optional
            Array of true labels for the events in these classification results
        indices: array_like of int, optional
            Array of indices of events to select out of the events in the fiTQun output (by default use all events).
        selection: index_expression, optional
            Selection to apply to the set of events to only use a subset of all events when plotting results, etc.
            (by default use all events).
        particle_label_map: dict
            Dictionary mapping particle type names to label integers. By default, use gamma:0, electron:1, muon:2, pi0:3
        plot_args: optional
            Additional arguments to pass to plotting functions, used to set the style when plotting these results
            together with other runs' results.
        """
        super().__init__(run_label=run_label, true_labels=true_labels, selection=selection, **plot_args)
        self.fitqun_output = fitqun_output
        if indices is None:
            indices = ...
        self.indices = indices
        if particle_label_map is None:
            particle_label_map = {'gamma': 0, 'electron': 1, 'muon': 2, 'pi0': 3}
        self.particle_label_map = particle_label_map
        self.gammas = {particle_label_map['gamma']}
        self.electrons = {particle_label_map['electron']}
        self.muons = {particle_label_map['muon']}
        self.pi0s = {particle_label_map['pi0']}
        self.electron_like = {self.particle_label_map[p] for p in ['electron', 'gamma']}
        self._electron_gamma_discriminator = None
        self._electron_muon_discriminator = None
        self._electron_pi0_nll_discriminator = None
        self._nll_pi0mass_discriminator = None
        self._electron_pi0_discriminator = None
        self._nll_pi0mass_factor = None

    def discriminator(self, signal_labels, background_labels):
        """
        Returns discriminator values given sets of labels representing the signal and background.
        For electron and/or gamma vs muon, use `electron_muon_discriminator`.
        For electron vs pi0, use `electron_pi0_discriminator`.
        For electron vs gamma, use `electron_gamma_discriminator`.
        No other combination of signal and background has currently been implemented for fiTQun results (other than
        swapping signal and background in any of the above cases, which returns the discriminator multiplied by -1).

        Parameters
        ----------
        signal_labels: int or sequence of ints
            Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
        background_labels: int or sequence of ints
            Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.

        Returns
        -------
        np.ndarray
            One-dimensional array of discriminator values, with length equal to the number of events in this run.
        """
        signal_labels = np.atleast_1d(signal_labels)
        background_labels = np.atleast_1d(background_labels)
        if set(signal_labels) <= self.electron_like and set(background_labels) <= self.muons:
            return self.electron_muon_discriminator
        elif set(signal_labels) <= self.muons and set(background_labels) <= self.electron_like:
            return self.muon_electron_discriminator
        elif set(signal_labels) <= self.electrons and set(background_labels) <= self.gammas:
            return self.electron_gamma_discriminator
        elif set(signal_labels) <= self.gammas and set(background_labels) <= self.electrons:
            return self.gamma_electron_discriminator
        elif set(signal_labels) <= self.electron_like and set(background_labels) <= self.pi0s:
            return self.electron_pi0_discriminator
        elif set(signal_labels) <= self.pi0s and set(background_labels) <= self.electron_like:
            return self.pi0_electron_discriminator
        else:
            raise NotImplementedError(f"A discriminator for the labels given for the signal {signal_labels} and "
                                      f"background {background_labels} has not yet been implemented for fiTQun outputs")

    def get_discriminator(self, discriminator):
        """
        Helper function for defining a particular discriminator. If `discriminator` is a function, it should take the
        fiTQun output as its only argument and return the discriminator, in which case the function called on this run's
        output is returned by this function. If `discriminator` is a string, it should name an attribute of this class
        to use as the discriminator, in which case that attribute is returned. In any other case the input is returned
        unchanged, for example if `discriminator` is already an array of discriminator values.

        Parameters
        ----------
        discriminator: callable or str or array_like of float

        Returns
        -------
        ndarray of float
            Array of discriminator values
        """
        if callable(discriminator):
            return discriminator(self.fitqun_output)[self.indices]
        elif isinstance(discriminator, str):
            return getattr(self, discriminator)
        else:
            return discriminator

    @property
    def electron_muon_discriminator(self):
        """Negative log-likelihood difference for electrons and muons: ln(L_e) - ln(L_mu)"""
        if self._electron_muon_discriminator is None:
            fq = self.fitqun_output
            self._electron_muon_discriminator = fq.muon_nll[self.indices] - fq.electron_nll[self.indices]
        return self._electron_muon_discriminator

    @property
    def muon_electron_discriminator(self):
        """Negative log-likelihood difference for electrons and muons: ln(L_mu) - ln(L_e)"""
        return -self.electron_muon_discriminator

    @property
    def electron_pi0_discriminator(self):
        """Discriminator for electron vs pi0, by default the log-likelihood difference: ln(L_e) - ln(L_pi0)"""
        if self._electron_pi0_discriminator is None:
            # By default, use simple discriminator using only the log-likelihood difference
            return self.electron_pi0_nll_discriminator
        return self._electron_pi0_discriminator

    @electron_pi0_discriminator.setter
    def electron_pi0_discriminator(self, discriminator):
        """Set the discriminator for electron vs pi0"""
        self._electron_pi0_discriminator = self.get_discriminator(discriminator)

    @property
    def electron_pi0_nll_discriminator(self):
        """Electron vs pi0 log-likelihood difference: ln(L_e) - ln(L_pi0)"""
        if self._electron_pi0_nll_discriminator is None:
            fq = self.fitqun_output
            self._electron_pi0_nll_discriminator = fq.pi0_nll[self.indices] - fq.electron_nll[self.indices]
        return self._electron_pi0_nll_discriminator

    @property
    def electron_pi0_nll_pi0mass_discriminator(self):
        """Linear 2D cut for electron vs pi0, in ln(L_e) - ln(L_pi0) and reconstructed pi0 mass"""
        if self._nll_pi0mass_discriminator is None:
            fq = self.fitqun_output
            self._nll_pi0mass_discriminator = (fq.pi0_nll[self.indices] - fq.electron_nll[self.indices]
                                               + self.nll_pi0mass_factor*fq.pi0_mass[self.indices])
        return self._nll_pi0mass_discriminator

    @property
    def nll_pi0mass_factor(self):
        """Gradient of the linear 2D cut in n(L_e) - ln(L_pi0) and reconstructed pi0 mass"""
        if self._nll_pi0mass_factor is None:
            self.tune_nll_pi0mass_discriminator()
        return self._nll_pi0mass_factor

    @nll_pi0mass_factor.setter
    def nll_pi0mass_factor(self, f):
        """Set the gradient of the linear 2D cut in n(L_e) - ln(L_pi0) and reconstructed pi0 mass"""
        self._nll_pi0mass_factor = f
        self._nll_pi0mass_discriminator = None  # Reset to be recalculated after changing factor

    def tune_nll_pi0mass_discriminator(self, pi0_efficiency=None, electron_efficiency=None, selection=None, binning=None,
                                       **opt_args):
        """
        Tune the gradient of the cut line for a linear 2D cut in n(L_e) - ln(L_pi0) and reconstructed pi0 mass.
        By default, optimize the gradient of the cut such that the Mann–Whitney U test is minimised. This minimises the
        sum of the ranks of the pi0 discriminator values when ranked together with the electron discriminator values.
        If `pi0_efficiency` is given, then the gradient is optimized to minimise the electron mis-PID when fixing a cut
        threshold that gives the desired pi0 efficiency.
        If `electron_efficiency` is given, then the gradient is optimized to minimise the pi0 mis-PID when fixing a cut
        threshold that gives the desired electron efficiency.
        If `binning` is provided, then the cut line gradient is tuned separately in each bin.

        Parameters
        ----------
        pi0_efficiency: float, optional
            Fixed pi0 efficiency for which to minimise electron mis-PID
        electron_efficiency: float, optional
            Fixed electron efficiency for which to minimise pi0 mis-PID
        selection: index_expression, optional
            If provided, only consider selected events when optimising the cut. By default, use the run's pre-defined
            selection, if any.
        binning: (np.ndarray, np.ndarray), optional
            Result of `analysis.utils.binning.get_binning` to use to tune the cut separately in each bin. By default,
            the cut is tuned once for all events without binning.
        opt_args: optional
            Additional arguments to pass to `scipy.optimize.minimize_scalar`

        Returns
        -------
        float or ndarray of floats
            The value of the optimal cut line gradient, or array of optimal cut line gradients in each bin if `binning`
            is provided.
        """
        if selection is None:
            selection = self.selection
        nll_diff = self.electron_pi0_nll_discriminator
        pi0mass = self.fitqun_output.pi0_mass[self.indices]
        electrons = np.isin(self.true_labels[selection], list(self.electrons))
        pi0s = np.isin(self.true_labels[selection], list(self.pi0s))

        if binning is not None:
            n_bins = len(binning[0]) + 1
            nll_pi0mass_factors = np.zeros(n_bins)
            for b in range(n_bins):
                bin_selection = np.zeros_like(self.true_labels, dtype=bool)
                bin_selection[selection] = True
                bin_selection &= (binning[1] == b)
                if np.any(bin_selection):
                    nll_pi0mass_factors[b] = self.tune_nll_pi0mass_discriminator(pi0_efficiency, electron_efficiency,
                                                                                 bin_selection, **opt_args)
            self.nll_pi0mass_factor = nll_pi0mass_factors
            self._nll_pi0mass_discriminator = nll_diff + nll_pi0mass_factors[binning[1]]*pi0mass
            self.electron_pi0_discriminator = self._nll_pi0mass_discriminator
            return nll_pi0mass_factors

        if pi0_efficiency is not None:  # Optimize cut to minimise electron mis-ID for given pi0 efficiency
            def e_misid(cut_gradient):
                discriminator = nll_diff[selection] + cut_gradient*pi0mass[selection]
                try:
                    cut_threshold = np.quantile(discriminator[pi0s], 1-pi0_efficiency)
                except IndexError as ex:
                    raise ValueError("There are zero selected pi0 events so cannot calculate a cut with any pi0 efficiency.") from ex
                return np.mean(discriminator[electrons] <= cut_threshold)
            min_func = e_misid
        elif electron_efficiency is not None:  # Optimize cut to minimise pi0 mis-ID for given electron efficiency
            def pi_misid(cut_gradient):
                discriminator = nll_diff[selection] + cut_gradient*pi0mass[selection]
                try:
                    cut_threshold = np.quantile(discriminator[electrons], 1-electron_efficiency)
                except IndexError as ex:
                    raise ValueError("There are zero selected electron events so cannot calculate a cut with any electron efficiency.") from ex
                return np.mean(discriminator[pi0s] > cut_threshold)
            min_func = pi_misid
        else:  # Optimise cut to minimise sum of ranks for pi0s (equivalent to Mann–Whitney U test)
            def u_test(cut_gradient):
                discriminator = nll_diff[selection] + cut_gradient*pi0mass[selection]
                ranks = np.argsort(discriminator)
                return np.sum(ranks[pi0s])
            min_func = u_test
        opt_args.setdefault('method', 'golden')
        result = minimize_scalar(min_func, **opt_args)
        self.nll_pi0mass_factor = result.x
        return self.nll_pi0mass_factor

    @property
    def pi0_electron_discriminator(self):
        """Discriminator for pi0 vs electron, by default the log-likelihood difference: ln(L_pi0) - ln(L_e)"""
        return -self.electron_pi0_discriminator

    @property
    def electron_gamma_discriminator(self):
        """
        Discriminator for electron vs gamma. The fiTQun gamma hypothesis doesn't work well, so by default the muon vs
        electron log-likelihood difference: ln(L_mu) - ln(L_gamma)
        """
        if self._electron_gamma_discriminator is None:
            #  fiTQun gamma hypotheses doesn't work well, so just use e/mu nll by default
            return self.muon_electron_discriminator
        return self._electron_gamma_discriminator

    @electron_gamma_discriminator.setter
    def electron_gamma_discriminator(self, discriminator):
        """Set the discriminator for electron vs gamma"""
        self._electron_gamma_discriminator = self.get_discriminator(discriminator)

    @property
    def gamma_electron_discriminator(self):
        """
        Discriminator for gamma vs electron. The fiTQun gamma hypothesis doesn't work well, so by default the electron
        vs gamma log-likelihood difference: ln(L_gamma) - ln(L_mu)
        """
        return -self.electron_gamma_discriminator
