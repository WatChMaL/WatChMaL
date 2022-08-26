import numpy as np
from analysis.utils.math import binomial_error


def get_binning(x, minimum, maximum, bins=200, width=None):
    if width is not None:
        bins = np.arange(minimum, maximum+width, width)
    elif isinstance(bins, int):
        bins = np.linspace(minimum, maximum, bins+1)
    return bins, np.digitize(x, bins)


def apply_binning(values, binning, selection=...):
    data = values[selection]
    data_bins = binning[1][selection]
    return [data[data_bins == b] for b in range(1, len(binning[0]))]


def binned_statistic(binned_values, function):
    return np.array([function(b) for b in binned_values])


def binned_resolutions(binned_values):
    return binned_statistic(binned_values, lambda y: np.quantile(y, 0.68))


def binned_std_errors(binned_values):
    return binned_statistic(binned_values, lambda y: np.std(y)/np.sqrt(len(y)))


def binned_binomial_errors(binned_values):
    return binned_statistic(binned_values, binomial_error)


def bin_centres(bins):
    return (bins[1:]+bins[:-1])/2


def bin_halfwidths(bins):
    return (bins[1:]-bins[:-1])/2
