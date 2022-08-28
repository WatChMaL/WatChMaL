import numpy as np
from analysis.utils.math import binomial_error


def get_binning(x, bins=None, minimum=None, maximum=None, width=None):
    """
    Finds the indices of the bins to which each value in input array belongs, for a set of bins specified either as an
    array of bin edges, number of bins or bin width

    Parameters
    ----------
    x: array_like
        Input array to be binned
    bins: array_like, optional
        If `bins` is an int, it defines the number of equal-width bins in the range (200, by default). If `bins` is an
        array, it is the array of bin edges and must be 1-dimensional and monotonic.
    minimum: int or real, optional
        Lowest bin lower edge (by default use minimum value in `x`). Not used if `bins` is an ndarray of bin edges.
    maximum: int or real, optional
        Highest bin upper edge (by default use minimum value in `x`). Not used if `bins` is an ndarray of bin edges.
    width: int or real, optional
        Width of bins to generate equal width bins if `bins` is None

    Returns
    -------
    bins: ndarray
        array of bin edges
    indices: ndarray
        output array of indices, of same shape as x
    """
    bin_array = np.array(bins)
    if bin_array.size == 1:
        if minimum is None:
            minimum = np.min(x)
        if maximum is None:
            maximum = np.max(x)
        if bins is None:
            bin_array = np.arange(minimum, maximum+width, width)
        else:
            bin_array = np.linspace(minimum, maximum, bins+1)
    indices = np.digitize(x, bin_array)
    return bin_array, indices


def apply_binning(values, binning, selection=...):
    """
    This function bins values according to the incides returned by `get_binning`. Returns a list of arrays where the nth
    array contains the values assigned to the nth bin.

    Parameters
    ----------
    values: array_like
        Values to be partitioned into bins

    binning: (ndarray, ndarray)
        Array of bin edges and array of bin indices, returned from `get_binning`

    selection: index expression, optional
        If provided, then `values` is indexed using this selection so that the values that pass the selection are binned
        and returned

    Returns
    -------
    list of ndarray
        List of arrays of values assigned to each bin
    """
    data = values[selection]
    data_bins = binning[1][selection]
    return [data[data_bins == b] for b in range(1, binning[0].size)]


def binned_resolutions(binned_residuals):
    """
    Calculate resolution defined as 68th percentile of the absolute residuals for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin

    Returns
    -------
    ndarray
        array of resolutions of the bins' residuals
    """
    return np.array([np.quantile(np.abs(x), 0.68) for x in binned_residuals])


def binned_std_errors(binned_residuals):
    """
    Calculate standard errors for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin

    Returns
    -------
    ndarray
        array of standard errors of the bins' residuals
    """
    return np.array([np.std(x)/np.sqrt(x.size) for x in binned_residuals])


def binned_binomial_errors(binned_results):
    """
    Calculate standard binomial errors for each of a list of arrays of binomial trial results

    Parameters
    ----------
    binned_results: list of array_like
        list of arrays of boolean binomial results in each bin

    Returns
    -------
    ndarray
        array of binomial errors of the bins' results
    """
    return np.array([binomial_error(x) for x in binned_results])


def bin_centres(bins):
    """Array of bin centres for an array of bin edges"""
    return (bins[1:]+bins[:-1])/2


def bin_halfwidths(bins):
    """Array of bin half-widths for an array of bin edges"""
    return (bins[1:]-bins[:-1])/2
