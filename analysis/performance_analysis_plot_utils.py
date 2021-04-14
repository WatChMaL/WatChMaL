"""
Utils for analyzing performance depending on specific variables
"""

import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from WatChMaL.analysis.performance_analysis_utils import compute_fixed_operating_performance, plot_fixed_operating_performance, compute_multi_var_fixed_operating_performance, plot_multi_var_fixed_operating_performance
from WatChMaL.analysis.performance_analysis_utils import compute_pion_fixed_operating_performance, plot_pion_fixed_operating_performance

# ========================================================================
# Single Variable Plotting Functions

def plot_fitqun_binned_performance(axes=None, **kwargs):
    '''
        Purpose: Re-create official FiTQun plots.
    '''
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(18,9), facecolor='w')

    plot_momentum_binned_performance(metric='efficiency', yrange=[0.7, 1.1], xrange=[0, 1000], plot_bins=None, ax=axes[0], **kwargs)

    plot_momentum_binned_performance(metric='fpr', yrange=[0, 0.01], xrange=[0, 1000], plot_bins=None, ax=axes[1], **kwargs)


def plot_single_var_binned_performance(scores_list, labels_list, reconstructed_momentum, plot_binning_features, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                            ax=None,marker='o--',colors=None,title_note='',metric='efficiency',yrange=None, xrange=None, names=None,
                            plot_bin_label=None, plot_bin_units=None, desired_labels=['$e$','$\mu$'], show_x_err=False):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        bin_centers, bin_metrics, yerr = compute_fixed_operating_performance(scores_list[idx], labels_list[idx], reconstructed_momentum, plot_binning_features,
                                                                             fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, metric, desired_labels)
        plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, 
                                    marker, colors[idx], 'Reconstructed Momentum', plot_bin_label, plot_bin_units,
                                    fpr_fixed_point, title_note, metric, 
                                    yrange, xrange, ax, show_x_err=show_x_err)
    ax.legend(names)


def plot_momentum_binned_performance(**kwargs):
    reconstructed_momentum = kwargs['reconstructed_momentum']
    recons_mom_bin_size    = kwargs['recons_mom_bin_size']
    plot_bins              = kwargs['plot_bins']
    
    if plot_bins is None:
        kwargs['plot_bins'] = np.array([0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))])
    
    return plot_single_var_binned_performance(plot_bin_label='Reconstructed Momentum', plot_bin_units='MeV/c', plot_binning_features=reconstructed_momentum, **kwargs)

def plot_true_momentum_binned_performance(momentum_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='True Momentum',  plot_bin_units='MeV/c', plot_binning_features=momentum_features, **kwargs)


def plot_energy_binned_performance(energy_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Energy', plot_bin_units='MeV/$c^2$', plot_binning_features=energy_features, **kwargs)


def plot_to_wall_binned_performance(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='To Wall',  plot_bin_units='cm', plot_binning_features=to_wall_features, **kwargs)


def plot_zenith_binned_performance(zenith_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Zenith',  plot_bin_units='Radians', plot_binning_features=zenith_features, **kwargs)


def plot_azimuth_binned_performance(azimuth_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Azimuth',  plot_bin_units='Radians$', plot_binning_features=azimuth_features, **kwargs)

# ========================================================================
# Multiple Variable Plotting Functions

def legend_without_duplicate_labels(ax):
    '''
    Merges legend elements with the same label (eliminates duplicate labels for models plotted with the same energy range)
    '''
    handles, labels = ax.get_legend_handles_labels()

    handle_dict = dict((k, []) for k in labels)
    for handle, label in zip(handles, labels):
        handle_dict[label].append(handle)

    new_handles = [tuple(h) for h in handle_dict.values()]

    ax.legend(new_handles, labels, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, prop={'size': 16}, bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_multi_var_binned_performance(
                            scores_list, labels_list, 
                            reconstructed_momentum, 
                            binning_features, 
                            binning_bin_size,
                            plot_binning_features,
                            index_dict, 
                            fixed_bin_label        = 'Reconstructed Momentum',
                            binning_bin_label      = 'Azimuth',
                            plot_bin_label         = 'Zenith',
                            fpr_fixed_point= 0.005, 
                            recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,
                            title_note = '',
                            metric='efficiency',yrange=None, xrange=None, names=None):
    
    title_note = ' in Bins of {}'.format(binning_bin_label) + title_note

    # TODO: fix cmap selection
    cmaps = ['Reds', 'Greens', 'Purples', 'Blues', 'Oranges',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        cmap = plt.cm.get_cmap(cmaps[idx])
        all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins = compute_multi_var_fixed_operating_performance(
                                                                                        scores                 = scores_list[idx], 
                                                                                        labels                 = labels_list[idx],
                                                                                        fixed_binning_features = reconstructed_momentum, 
                                                                                        fixed_bin_size         = 50,
                                                                                        binning_features       = binning_features,
                                                                                        binning_bin_size       = binning_bin_size,
                                                                                        plot_binning_features  = plot_binning_features, 
                                                                                        plot_bins              = plot_bins,
                                                                                        index_dict             = index_dict, 
                                                                                        ignore_dict            = index_dict,
                                                                                        muon_comparison        = False, 
                                                                                        use_rejection          = False,
                                                                                        metric                 = 'efficiency',
                                                                                        fpr_fixed_point        = fpr_fixed_point)

        plot_multi_var_fixed_operating_performance(all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins, 
                                                   marker, cmap, 
                                                   fixed_bin_label, binning_bin_label, plot_bin_label, 
                                                   fpr_fixed_point, title_note, metric, yrange, xrange, axes)

    legend_without_duplicate_labels(axes)
    
    #if names is not None:
        #axes.legend(names)
    

def plot_to_wall_binned_in_energy(to_wall_features, **kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='To Wall', plot_bin_label='To Wall', plot_binning_features=to_wall_features, **kwargs)


def plot_zenith_binned_in_azimuth(zenith_features, **kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='Azimuth', plot_bin_label='Zenith', plot_binning_features=zenith_features, **kwargs)


def plot_azimuth_binned_in_zenith(azimuth_features, **kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='Zenith', plot_bin_label='Azimuth', plot_binning_features=azimuth_features, **kwargs)


# ========================================================================
# Define equivalent plotting functions for pions

def plot_single_var_pion_binned_performance(
                                            scores, labels, 
                                            fixed_binning_features, fixed_bin_label, 
                                            plot_binning_features, plot_bin_label,
                                            p0, p1, pi0mass,
                                            fpr_fixed_point, index_dict, fixed_bin_size=50, plot_bins=20, 
                                            marker='o--',color='k',title_note='', metric='efficiency',yrange=None,xrange=None,
                                            ax = None, show_x_err=True, publication=False):

    bin_centers, bin_metrics, yerr = compute_pion_fixed_operating_performance(
                                     scores, labels, 
                                     fixed_binning_features,
                                     plot_binning_features,
                                     p0, p1, pi0mass,
                                     fpr_fixed_point, index_dict, fixed_bin_size, plot_bins, 
                                     metric)

    plot_pion_fixed_operating_performance(bin_centers, bin_metrics, yerr,
                                    fixed_bin_label, 
                                     plot_bin_label,
                                     fpr_fixed_point, index_dict, fixed_bin_size, plot_bins, 
                                     marker,color,title_note, metric,yrange,xrange,
                                     ax, show_x_err, publication)
# ========================================================================
# Define helper functions

def plot_joint_legend(fig_list):
    return -1