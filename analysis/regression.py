import numpy as np
import analysis.utils.binning as bins
import matplotlib.pyplot as plt


def get_predictions(run_directory, indices):
    predictions = np.load(run_directory + "/outputs/predictions.npy")
    output_indices = np.load(run_directory + "/outputs/indices.npy")
    intersection = np.intersect1d(indices, output_indices, return_indices=True)
    sorted_predictions = np.zeros(indices.shape + predictions.shape[1:])
    sorted_predictions[intersection[1]] = predictions[intersection[2]]
    return sorted_predictions.squeeze()


def plot_histograms(runs, quantity, selection=..., figsize=(12, 9), xlabel="", ylabel="", legend='best', tight=True,
                    **hist_args):
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


def plot_resolution_profile(runs, quantity, binning, selection=..., figsize=(12, 9), xlabel="", ylabel="",
                            legend='best', tight=True, ylim=None, **plot_args):
    fig, ax = plt.subplots(figsize=figsize)
    for r in runs:
        args = {**plot_args, **r['args']}
        plot_binned_resolution(r[quantity], binning, ax, selection, **args)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend)
    if tight:
        fig.tight_layout()
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def plot_binned_resolution(values, binning, ax, selection=..., errors=False, xerrors=True, **plot_args):
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
