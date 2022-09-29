"""
Utility functions for binning events in some quantity and manipulating other quantities based on the binning
"""
import numpy as np
from analysis.utils.math import binomial_error


def get_binning(x, bins=None, minimum=None, maximum=None, width=None):
    """
    Finds the indices of the bins to which each value in input array belongs, for a set of bins specified either as an
    array of bin edges, number of bins or bin width

    Parameters
    ----------
    x: array_like
        Input array to be binned.
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
    bins: np.ndarray
        array of bin edges
    indices: np.ndarray
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
    This function bins values according to the indices returned by `get_binning`. Returns a list of arrays where the nth
    array contains the values assigned to the nth bin.

    Parameters
    ----------
    values: array_like
        Values to be partitioned into bins

    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `get_binning`

    selection: index expression, optional
        If provided, then `values` is indexed using this selection so that the values that pass the selection are binned
        and returned

    Returns
    -------
    list of np.ndarray
        List of arrays of values assigned to each bin
    """
    data = values[selection]
    data_bins = binning[1][selection]
    return [data[data_bins == b] for b in range(1, binning[0].size)]


def unapply_binning(binned_values, binning, selection=...):
    """
    Reverses the effect of `apply_binning`. Takes a list of arrays corresponding to values assigned to bins and returns
    a single array of values ordered as they were before they were binned.

    Parameters
    ----------
    binned_values: list of np.ndarray
        Binned values returned by `apply_binning`

    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `get_binning`

    selection: index expression, optional
        If provided, the returned array will match the original length before the selection of `apply_binning` was
        applied. The missing entries that do not exist in `binned_values` due to not passing the selection are filled
        with zeros.

    Returns
    -------
    list of np.ndarray
        List of arrays of values assigned to each bin
    """
    data = np.zeros(binning[1].shape)
    data_bins = binning[1][selection]
    for b, v in enumerate(binned_values):
        data[selection][data_bins == b+1] = v


def binned_resolutions(binned_residuals, return_errors=True):
    """
    Calculate resolution defined as 68th percentile of the absolute residuals for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin
    return_errors: bool
        if True, return array of standard error on the mean of each list of values (as a proxy for the standard error on
        each list's 68th percentile).

    Returns
    -------
    resolutions: np.ndarray
        array of resolutions of the bins' residuals
    errors: np.ndarray
        array of standard errors on the means of the residuals
    """
    resolutions = np.array([np.quantile(np.abs(x), 0.68) for x in binned_residuals])
    if return_errors:
        errors = binned_std_errors(binned_residuals)
        return resolutions, errors
    else:
        return resolutions


def binned_quantiles(binned_values, quantile):
    """
    Calculate quantiles of the values for each of a list of arrays of values

    Parameters
    ----------
    binned_values: list of array_like
        list of arrays of float values in each bin
    quantile: float
        quantile value to find in each bin

    Returns
    -------
    np.ndarray
        array of quantiles of the bins' values
    """
    return np.array([np.quantile(x, quantile) for x in binned_values])


def binned_mean(binned_values, return_errors=True):
    """
    Calculate mean of the values for each of a list of arrays of values

    Parameters
    ----------
    binned_values: list of array_like
        list of arrays of values in each bin
    return_errors: bool
        if True, return array of standard error on the mean of each list of values

    Returns
    -------
    means: np.ndarray
        array of means of the bins' values
    errors: np.ndarray, optional
        array of standard errors of the means
    """
    means = np.array([np.mean(x) for x in binned_values])
    if return_errors:
        errors = binned_std_errors(binned_values)
        return means, errors
    else:
        return means


def binned_efficiencies(binned_cut, return_errors=True, reverse=False):
    """
    Calculate percentage of true values (and binomial errors) in each arrays of booleans in a list

    Parameters
    ----------
    binned_cut: list of array_like
        list of arrays of booleans in each bin
    reverse: bool
        If True, reverse the cut to give percentage of events failing the cut. By default, give percentage of events
        passing the cut
    return_errors: bool
        if True, return array of each list of booleans' binomial standard error

    Returns
    -------
    efficiencies: np.ndarray
        array of percentage of true values in each bin
    errors: np.ndarray, optional
        array of binomial standard errors
    """
    efficiencies = binned_mean(binned_cut, return_errors=False)*100
    if reverse:
        efficiencies = 100 - efficiencies
    if return_errors:
        errors = binned_binomial_errors(binned_cut) * 100
        return efficiencies, errors
    else:
        return efficiencies


def binned_std_errors(binned_residuals):
    """
    Calculate standard errors for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin

    Returns
    -------
    np.ndarray
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
    np.ndarray
        array of binomial errors of the bins' results
    """
    return np.array([binomial_error(x) for x in binned_results])


def bin_centres(bins):
    """Array of bin centres for an array of bin edges"""
    return (bins[1:]+bins[:-1])/2


def bin_halfwidths(bins):
    """Array of bin half-widths for an array of bin edges"""
    return (bins[1:]-bins[:-1])/2
