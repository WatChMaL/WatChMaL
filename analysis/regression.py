import numpy as np
import analysis.utils.binning as bins
import matplotlib.pyplot as plt
import tabulate


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


def plot_histograms(runs, quantity, selection=..., fig_size=None, x_label="", y_label="", legend='best', **hist_args):
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
    fig_size: (float, float), optional
        Figure size.
    x_label: str, optional
        Label of the x-axis.
    y_label: str, optional
        Label of the y-axis.
    legend: str or None, optional
        Position of the legend, or None to have no legend. Attempts to find the best position by default.
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
    fig, ax = plt.subplots(figsize=fig_size)
    for r in runs:
        data = r[quantity][selection].flatten()
        args = {**hist_args, **r['args']}
        ax.hist(data, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    return fig, ax


def plot_resolution_profile(runs, quantity, binning, selection=..., fig_size=None, x_label="", y_label="",
                            legend='best', y_lim=None, **plot_args):
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
        plot_binned_resolution(r[quantity], ax, binning, selection, **args)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.legend(loc=legend)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax


def plot_binned_resolution(values, ax, binning, selection=..., errors=False, x_errors=True, **plot_args):
    """
    Plot binned resolutions of the results of a regression run on an existing set of axes. The values should correspond
    to residuals, and the set of residuals are divided up into bins of some quantity according to `binning`, before
    calculating the resolution (68th percentile of their absolute values) in each bin. A selection can be provided to
    use only a subset of all the values.

    Parameters
    ----------
    values: array_like
        Array of residuals to be binned and their corresponding resolutions plotted.
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
    binned_values = bins.apply_binning(values, binning, selection)
    y = bins.binned_resolutions(binned_values)
    x = bins.bin_centres(binning[0])
    if errors:
        y_err = bins.binned_std_errors(binned_values)
        x_err = bins.bin_halfwidths(binning[0]) if x_errors else None
        plot_args.setdefault('marker', '')
        plot_args.setdefault('capsize', 4)
        plot_args.setdefault('capthick', 2)
        ax.errorbar(x, y, yerr=y_err, xerr=x_err, **plot_args)
    else:
        plot_args.setdefault('marker', 'o')
        ax.plot(x, y, **plot_args)


def get_resolutions(runs, quantity, selection=...):
    """
    Return a list of resolutions (68th percentile) of values as some quantity in each run.

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be used and the key "args"
        containing a dictionary of arguments including "label" to label the run.
    quantity: str
        Key in `runs` that contains the quantities whose resolutions would be calculated.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).

    Returns
    -------
    list of float
        List of resolution of quantity of each run.
    """
    return [np.quantile(np.abs(r[quantity][selection]), 0.68) for r in runs]


def get_means(runs, quantity, selection=...):
    """
    Return a list of means of values as some quantity in each run.

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be used and the key "args"
        containing a dictionary of arguments including "label" to label the run.
    quantity: str
        Key in `runs` that contains the quantities whose means would be calculated.
    selection: indexing expression, optional
        Selection of the values to use in calculating the means (by default use all values).

    Returns
    -------
    list of float
        List of mean quantity of each run.
    """
    return [np.mean(r[quantity][selection]) for r in runs]


def tabulate_statistics(runs, quantities, labels, selection=..., statistic="resolution", transpose=False,
                        **tabulate_args):
    """
    Return a table of summary statistics of quantities of runs.

    Parameters
    ----------
    runs: dict
        Dictionary of run results, with the key "quantity" giving an array-like of values to be used and the key "args"
        containing a dictionary of arguments including "label" to label the run.
    quantities: str or list of str
        Key in `runs` that contains the quantities whose statistics would be calculated, or list of keys.
    labels: str or list of str
        Label for the quantities / statistics being calculated, or list of labels the same length as `quantities`.
    selection:
        Selection of the values to use in calculating the summary statistics (by default use all values).
    statistic: {callable, 'resolution', 'mean'} or list of {callable, 'resolution', 'mean'}
        The summary statistic to apply to the quantity. If callable, should be a function that takes the array_like of
        values and returns the summary statistic. If `resolution` (default) use the 68th percentile. If `mean` use the
        mean. If a list, should be the same length as `quantities` to specify the summary statistic of each quantity.
    transpose: bool
        If True, table rows correspond to each run and columns correspond to each quantity summary statistic. Otherwise
        (default) rows correspond to summary statistics and columns correspond to runs.
    tabulate_args: optional
        Additional named arguments to pass to `tabulate.tabulate`. By default, set table format to `html` and float
        format to `.2f`.
    Returns
    -------
    str
        String representing the tabulated data
    """
    tabulate_args.setdefault('tablefmt', 'html')
    tabulate_args.setdefault('floatfmt', '.2f')
    if isinstance(quantities, str):
        quantities = [quantities]
    if isinstance(labels, str):
        labels = [labels]
    statistic_map = {
        "resolution": get_resolutions,
        "mean": get_means,
    }
    if callable(statistic):
        functions = [lambda rs, q, sel: [statistic(r[q][sel]) for r in rs]] * len(quantities)
    elif isinstance(statistic, str):
        functions = [statistic_map[statistic]]*len(quantities)
    else:
        functions = [(lambda rs, q, sel: [s(r[q][sel]) for r in rs]) if callable(s)
                     else statistic_map[s]
                     for s in statistic]
    data = []
    for f, q in zip(functions, quantities):
        data.append(f(runs, q, selection))
    if transpose:
        data = list(zip(*data))
        headers = labels
        labels = [r['args']['label'] for r in runs]
    else:
        headers = [r['args']['label'] for r in runs]
    return tabulate.tabulate(data, headers=headers, showindex=labels, **tabulate_args)
