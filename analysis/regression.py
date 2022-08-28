import numpy as np
import analysis.utils.binning as bins
import matplotlib.pyplot as plt


def get_predictions(run_directory, indices=None):
    """
    Read the predictions resulting from an evaluation run of a WatChMaL regression model.

    Parameters
    ----------
    run_directory: str
        Top-level output directory of a WatChMaL regression run.
    indices: array_like of int, optional
        array of indices of predictions to select out of the indices output by WatChMaL (by default return all
        predictions sorted by their indices).

    Returns
    -------
    ndarray
        Array of predictions.
    """
    predictions = np.load(run_directory + "/outputs/predictions.npy")
    output_indices = np.load(run_directory + "/outputs/indices.npy")
    if indices is None:
        return predictions[output_indices.argsort()].squeeze()
    intersection = np.intersect1d(indices, output_indices, return_indices=True)
    sorted_predictions = np.zeros(indices.shape + predictions.shape[1:])
    sorted_predictions[intersection[1]] = predictions[intersection[2]]
    return sorted_predictions.squeeze()


def plot_histograms(runs, quantity, selection=..., figsize=None, xlabel="", ylabel="", legend='best', tight=True,
                    **hist_args):
    """
    Plot overlaid histograms of results from a number of regression runs

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be histogrammed and the key
        "args" containing a dictionary of arguments to the `hist` plotting function.
    quantity: str
        Key in `runs` that contains the quantities to be histogrammed
    selection: indexing expression, optional
        Selection of the values to be histogrammed (by default use all values).
    figsize: (float, float), optional
        Figure size.
    xlabel: str, optional
        Label of the x-axis.
    ylabel: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    tight: bool, optional
        If false, don't use tight formatting of the figure.
    hist_args: optional
        Additional arguments to pass to the `hist` plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: Figure
    ax: axes.Axes
    """
    hist_args.setdefault('bins', 200)
    hist_args.setdefault('density', True)
    hist_args.setdefault('histtype', 'step')
    hist_args.setdefault('lw', 2)
    fig, ax = plt.subplots(figsize=figsize)
    for r in runs:
        data = r[quantity][selection].flatten()
        args = {**hist_args, **r['args']}
        ax.hist(data, **args)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend)
    if tight:
        fig.tight_layout()
    return fig, ax


def plot_resolution_profile(runs, quantity, binning, selection=..., figsize=None, xlabel="", ylabel="", legend='best',
                            tight=True, ylim=None, **plot_args):
    """
    Plot binned resolutions for results from a number of regression runs.
    The quantity being used from each run should correspond to residuals and these residuals are divided up into bins of
    some quantity according to `binning`, before calculating the resolution (68th percentile of their absolute values)
    in each bin. A selection can be provided to use only a subset of all the values. The same binning and selection is
    applied to each run.

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be used and the key "args"
        containing a dictionary of arguments to the plotting function.
    quantity: str
        Key in `runs` that contains the quantities whose resolutions would be plotted.
    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    figsize: (float, float), optional
        Figure size.
    xlabel: str, optional
        Label of the x-axis.
    ylabel: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
    tight: bool, optional
        If false, don't use tight formatting of the figure.
    ylim: (float, float), optional
        Limits of the y-axis.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.

    Returns
    -------
    fig: Figure
    ax: axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    for r in runs:
        args = {**plot_args, **r['args']}
        plot_binned_resolution(r[quantity], ax, binning, selection, **args)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend)
    if tight:
        fig.tight_layout()
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def plot_binned_resolution(values, ax, binning, selection=..., errors=False, xerrors=True, **plot_args):
    """
    Plot binned resolutions of the results of a regression run on an existing set of axes. The values should correspond
    to residuals, and the set of residuals are divided up into bins of some quantity according to `binning`, before
    calculating the resolution (68th percentile of their absolute values) in each bin. A selection can be provided to
    use only a subset of all the values.

    Parameters
    ----------
    values: array_like
        Array of residuals to be binned and the.
    ax: axes.Axes
        Axes to draw the plot.
    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    errors: bool, optional
        If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
    xerrors: bool, optional
        If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.
    """
    plot_args.setdefault('lw', 2)
    binned_values = bins.apply_binning(values, binning, selection)
    y = bins.binned_resolutions(binned_values)
    x = bins.bin_centres(binning[0])
    if errors:
        yerr = bins.binned_std_errors(binned_values)
        xerr = bins.bin_halfwidths(binning[0]) if xerrors else None
        plot_args.setdefault('marker', '')
        plot_args.setdefault('capsize', 4)
        plot_args.setdefault('capthick', 2)
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, **plot_args)
    else:
        plot_args.setdefault('marker', 'o')
        ax.plot(x, y, **plot_args)
