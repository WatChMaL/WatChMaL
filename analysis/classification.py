import numpy as np
import analysis.utils.binning as bins
import analysis.utils.plotting as plot
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
import tabulate


def get_softmaxes(run_directory, indices=None):
    """
    Read the softmax predictions resulting from an evaluation run of a WatChMaL classification model.

    Parameters
    ----------
    run_directory: str
        Top-level output directory of a WatChMaL classification run.
    indices: array_like of int, optional
        array of indices of softmaxes to select out of the indices output by WatChMaL (by default return all softmaxes
        sorted by their indices).

    Returns
    -------
    ndarray
        Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
        contains the softmax values of a class.
    """
    softmaxes = np.load(run_directory + "/outputs/softmax.npy")
    output_indices = np.load(run_directory + "/outputs/indices.npy")
    if indices is None:
        return softmaxes[output_indices.argsort()].squeeze()
    intersection = np.intersect1d(indices, output_indices, return_indices=True)
    sorted_softmaxes = np.zeros(indices.shape + softmaxes.shape[1:])
    sorted_softmaxes[intersection[1]] = softmaxes[intersection[2]]
    return sorted_softmaxes.squeeze()


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


def softmax_discriminator(softmaxes, signal_labels, background_labels):
    """
    Return a discriminator with appropriate scaling of softmax values from multi-class training, given the set of signal
    and background class labels. For each event, the discriminator is the sum the signal softmax values normalised by
    the sum of signal and background softmax values.

    Parameters
    ----------
    softmaxes: ndarray
        Two dimensional array of predicted softmax values, where each row corresponds to an event and each column
        contains the softmax values of a class.
    signal_labels: int or sequence of ints
        Set of labels corresponding to signal classes. Can be either a single label or a sequence of labels.
    background_labels: int or sequence of ints
        Set of labels corresponding to background classes. Can be either a single label or a sequence of labels.

    Returns
    -------
    ndarray
        One-dimensional array of discriminator values, with length equal to the first dimension of `softmaxes`.
    """
    signal_softmax = combine_softmax(softmaxes, signal_labels)
    background_softmax = combine_softmax(softmaxes, background_labels)
    return signal_softmax/(signal_softmax+background_softmax)


def plot_rocs(runs, discriminator, signal, selection=..., fig_size=None, x_label="", y_label="", x_lim=None, y_lim=None,
              y_log=None, x_log=None, legend='best', mode='rejection', **plot_args):
    """
    Plot overlaid ROC curves of results from a number of classification runs

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key given by `discriminator` giving an array-like of discriminator values
        to use and the key "args" containing a dictionary of arguments to the `matplotlib.pyplot.plot` plotting function.
    discriminator: str
        Key in `runs` that contains the discriminator values to use for the ROC curve.
    signal: array_like of bools
        One dimensional array of boolean values, the same length as the discriminator array in each run, indicating
        whether each event is classed as signal.
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
        provided in `runs`.

    Returns
    -------
    fig: Figure
    ax: axes.Axes
    """
    fig, ax = plt.subplots(figsize=fig_size)
    selected_signal = signal[selection]
    for r in runs:
        fpr, tpr, _ = metrics.roc_curve(selected_signal, r[discriminator][selection])
        auc = metrics.auc(fpr, tpr)
        args = {**plot_args, **r['args']}
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


def cut_with_constant_binned_efficiency(discriminator_values, efficiency, binning, selection=...,
                                        return_thresholds=False):
    """
    Generate array of boolean values indicating whether each event passes a cut defined such that, in each bin of some
    binning of the events, a constant proportion of the selected events pass the cut.
    After taking the subset of `discriminator_values` defined by `selection`, in each bin of `binning` the threshold
    discriminator value is found such that the proportion that are above the threshold is equal to `efficiency`. These
    cut thresholds are then used to apply the cut to all events (not just those selected by `selection`) and an array of
    booleans is returned for whether each discriminator value is above the threshold of its corresponding bin.

    Parameters
    ----------
    discriminator_values: array_like
        One dimensional array of discriminator values to use to generate the cut.
    efficiency: float
        The fixed efficiency to ensure in each bin.
    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the discriminator values to use in calculating the thresholds applied by the cut in each bin (by
        default use all values).
    return_thresholds:
        If True, return also the array of cut thresholds calculated for each bin.

    Returns
    -------
    cut: ndarray of bool
        One-dimensional array the same length as `discriminator_values` indicating whether each event passes the cut.
    thresholds: ndarray of float
        One-dimensional array giving the threshold applied by the cut to events in each bin.
    """
    binned_discriminators = bins.apply_binning(discriminator_values, binning, selection)
    thresholds = bins.binned_quantiles(binned_discriminators, 1 - efficiency)
    # put inf as first and last threshold for overflow bins
    padded_thresholds = np.concatenate(([np.inf], thresholds, [np.inf]))
    cut = np.array(discriminator_values) > padded_thresholds[binning[1]]
    if return_thresholds:
        return cut, thresholds
    else:
        return cut


def plot_binned_efficiency(cut, ax, binning, selection=..., errors=False, x_errors=True, **plot_args):
    """
    Plot binned efficiencies of a cut applied to a classification run on an existing set of axes. The cut values should
    correspond to booleans indicating whether each event passes the cut, then the set of booleans are divided up into
    bins of some quantity according to `binning`, before calculating the efficiency (proportion of events passing the
    cut) in each bin. A selection can be provided to use only a subset of all the values.

    Parameters
    ----------
    cut: array_like of bool
        Array of booleans indicating whether each event passes the cut.
    ax: axes.Axes
        Axes to draw the plot.
    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    errors: bool, optional
        If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
    x_errors: bool, optional
        If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.
    """
    plot_args.setdefault('lw', 2)
    binned_cut = bins.apply_binning(cut, binning, selection)
    y = bins.binned_efficiencies(binned_cut, errors)
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


def plot_efficiency_profile(runs, cut, binning, selection=..., fig_size=None, x_label="", y_label="", legend='best',
                            y_lim=None, **plot_args):
    """
    Plot binned efficiencies for a cut applied to a number of classification runs.
    The cut applied to each run should correspond to booleans indicating whether each event passes the cut, then in each
    bin the proportion of events passing the cut is calculated as the efficiency and plotted.
    A selection can be provided to use only a subset of all the values. The same binning and selection is applied to
    each run.

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be used and the key "args"
        containing a dictionary of arguments to the plotting function.
    cut: str
        Key in `runs` that contains the booleans indicating which events pass the cut, to calculate the efficiencies.
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
        args = {**plot_args, **r['args']}
        plot_binned_efficiency(r[cut], ax, binning, selection, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


def load_training_log(run_directory):
    log_val = np.genfromtxt(run_directory+"/outputs/log_val.csv", delimiter=',',skip_header=1)
    val_iteration = log_val[:, 0]
    val_loss = log_val[:, 1]
    val_accuracy = log_val[:, 2]
    val_best = log_val[:, 3].astype(bool)
    log_train = np.array([
        np.genfromtxt(f, delimiter=',', skip_header=1)
        for f in glob.glob(run_directory+"/outputs/log_train*.csv")
    ])
    train_iteration = log_train[0, :, 0]
    train_epoch = log_train[0, :, 1]
    train_loss = np.mean(log_train[:, :, 2], axis=0)
    train_accuracy = np.mean(log_train[:, :, 3], axis=0)
    it_per_epoch = np.min(train_iteration[train_epoch == 1]) - 1
    train_epoch = train_iteration / it_per_epoch
    val_epoch = val_iteration / it_per_epoch
    return train_epoch, train_loss, train_accuracy, val_epoch, val_loss, val_accuracy, val_best


def plot_training_progression(train_epoch, train_loss, train_accuracy, val_epoch, val_loss, val_accuracy,
                                             val_best=None, y_loss_lim=None, fig_size=None, title=None, legend='center right'):
    fig, ax1 = plot.plot_training_progression(train_epoch, train_loss, val_epoch, val_loss, val_best, y_loss_lim, fig_size, title, legend=None)
    ax2 = ax1.twinx()
    ax2.plot(train_epoch, train_accuracy, lw=2, label='Train accuracy', color='r', alpha=0.3)
    ax2.plot(val_epoch, val_accuracy, lw=2, label='Validation accuracy', color='r')
    if val_best is not None:
        ax2.plot(val_epoch[val_best], val_accuracy[val_best], lw=0, marker='o', label='Best validation accuracy',
                 color='darkred')
    ax2.set_ylabel("Accuracy", c='r')
    if legend:
        ax1.legend(plot.combine_legends((ax1, ax2)), loc=legend)
    return fig, ax1, ax2
