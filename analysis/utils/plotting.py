"""
Utility functions for plotting
"""

import matplotlib
from matplotlib import pyplot as plt
import analysis.utils.binning as bins


def combine_legends(ax):
    """
    Combine legend entries from multiple axes.

    Parameters
    ----------
    ax: sequence of matplotlib.axes.Axes
        The axes whose legend entries should be combined.

    Returns
    -------
    (list, list):
        Handles and labels of the combined legend.
    """
    legends = [a.get_legend_handles_labels() for a in ax]
    return (l1 + l2 for l1, l2 in zip(*legends))


def plot_legend(ax):
    """
    Plot a standalone legend for the entries plotted in one or multiple axes.

    Parameters
    ----------
    ax: matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        The axes whose legend entries should be plotted.

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axes
    """
    if isinstance(ax, matplotlib.axes.Axes):
        leg_params = ax.get_legend_handles_labels()
    else:
        leg_params = combine_legends(ax)
    leg_fig, leg_ax = plt.subplots(figsize=(1, 1))
    leg_ax.axis(False)
    leg_fig.set_tight_layout(False)
    leg_fig.legend(*leg_params, loc='center')
    return leg_fig, leg_ax


def plot_binned_values(ax, func, values, binning, selection=None, errors=False, x_errors=True, **plot_args):
    """
    Plot a binned statistic for some values on an existing set of axes.
    The values are divided up into bins of some quantity according to `binning`, with some statistic function applies to
    the values in each bin. The results of the statistic and optionally error bars (if errors are provided by the
    statistic function) for each bin are plotted against the binning quantity on the x-axis. A selection can be provided
    to use only a subset of all the values.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes to draw the plot.
    func: callable
        A function that takes the values as its first parameter and a boolean for whether to return errors as its second
        parameter and returns the binned results and optional errors.
    values: array_like
        Array of values to be binned and passed to `func`.
    binning: (np.ndarray, np.ndarray)
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
    binned_values = bins.apply_binning(values, binning, selection)
    x = bins.bin_centres(binning[0])
    if errors:
        y_values, y_errors = func(binned_values, errors)
        x_errors = bins.bin_halfwidths(binning[0]) if x_errors else None
        plot_args.setdefault('marker', '')
        plot_args.setdefault('capsize', 4)
        plot_args.setdefault('capthick', 2)
        ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, **plot_args)
    else:
        y = func(binned_values, errors)
        plot_args.setdefault('marker', 'o')
        ax.plot(x, y, **plot_args)
