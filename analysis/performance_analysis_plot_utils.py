"""
Utils for analyzing performance depending on specific variables
"""
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from WatChMaL.analysis.performance_analysis_utils import compute_fixed_operating_performance, plot_fixed_operating_performance, compute_multi_var_fixed_operating_performance

# ========================================================================
# Single Variable Plotting Functions

"""
def plot_fitqun_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                                    axes=None,marker='o--',colors=None,title_note='',names=None):
    '''
        Purpose: Re-create official FiTQun plots.
        Args:
            scores                  ... network scores for each class
            labels                  ... 1d array of labels
            true_momentum           ... 1d array of event true momentum
            reconstructed_momentum  ... 1d array of FQ reconstructed momentum
            fpr_fixed_point         ... fixed false-positive rate for FQ recons. mom. bins
            index_dict              ... dictionary with 'e', 'mu' keys pointing to corresponding integer labels
            recons_mom_bin_size     ... size of reconstructed mom. bin
            true_mom_bins           ... number of true momentum bins
            ax                      ... axis to plot on
            marker                  ... marker for plor
            color                   ... curve color
            title_note              ... string to append to title
            metric                  ... 'efficiency' will give signal efficiency, any other will give FPR
            yrange                  ... range for the y axis
    '''
    
    #if axes is None:
    fig, axes = plt.subplots(1,2,figsize=(18,9), facecolor='w')

    plot_momentum_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, 
                                     axes[0],marker,colors,title_note,'efficiency',yrange=[0.7, 1.1],xrange=[0, 1000],names=names)

    plot_momentum_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, 
                                     axes[1],marker,colors,title_note,'fpr',yrange=[0, 0.01],xrange=[0, 1000],names=names)
"""


def plot_fitqun_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                                    axes=None,marker='o--',colors=None,title_note='',names=None):
    '''
        Purpose: Re-create official FiTQun plots.
        Args:
            scores                  ... network scores for each class
            labels                  ... 1d array of labels
            true_momentum           ... 1d array of event true momentum
            reconstructed_momentum  ... 1d array of FQ reconstructed momentum
            fpr_fixed_point         ... fixed false-positive rate for FQ recons. mom. bins
            index_dict              ... dictionary with 'e', 'mu' keys pointing to corresponding integer labels
            recons_mom_bin_size     ... size of reconstructed mom. bin
            true_mom_bins           ... number of true momentum bins
            ax                      ... axis to plot on
            marker                  ... marker for plor
            color                   ... curve color
            title_note              ... string to append to title
            metric                  ... 'efficiency' will give signal efficiency, any other will give FPR
            yrange                  ... range for the y axis
    '''
    
    #if axes is None:
    fig, axes = plt.subplots(1,2,figsize=(18,9), facecolor='w')

    plot_momentum_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, 
                                     axes[0],marker,colors,title_note,'efficiency',yrange=[0.7, 1.1],xrange=[0, 1000],names=names)

    plot_momentum_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, 
                                     axes[1],marker,colors,title_note,'fpr',yrange=[0, 0.01],xrange=[0, 1000],names=names)


def plot_single_var_binned_performance(scores_list, labels_list, reconstructed_momentum, plot_binning_features, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,title_note='',metric='efficiency',yrange=None, xrange=None, names=None,
                            plot_bin_label=None):
    
    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        bin_centers, bin_metrics, yerr = compute_fixed_operating_performance(scores_list[idx], labels_list[idx], reconstructed_momentum, plot_binning_features,
                                                                             fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, metric)
        plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, 
                                    marker, colors[idx], 'Reconstructed Momentum', plot_bin_label, 
                                    fpr_fixed_point, title_note, metric, 
                                    yrange, xrange, axes)
    axes.legend(names)

"""
def plot_momentum_binned_performance(scores_list, labels_list, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,title_note='',metric='efficiency',yrange=None, xrange=None, names=None):
    
    #assert reconstructed_momentum.shape[0] == scores.shape[0], 'Error: reconstructed_momentum must have same length as softmaxes'
    
    if plot_bins is None:
        plot_bins = np.array([0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))])

    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        bin_centers, bin_metrics, yerr = compute_fixed_operating_performance(scores_list[idx], labels_list[idx], reconstructed_momentum, reconstructed_momentum,
                                                                             fpr_fixed_point, index_dict, recons_mom_bin_size, plot_bins, metric)
        plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, 
                                    marker, colors[idx], 'Reconstructed Momentum', 'Reconstructed Momentum', 
                                    fpr_fixed_point, title_note, metric, 
                                    yrange, xrange, axes)
    axes.legend(names)
"""

def plot_momentum_binned_performance(**kwargs):
    reconstructed_momentum = kwargs['reconstructed_momentum']
    recons_mom_bin_size    = kwargs['recons_mom_bin_size']
    plot_bins              = kwargs['plot_bins']
    
    if plot_bins is None:
        kwargs['plot_bins'] = np.array([0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))])
    
    return plot_single_var_binned_performance(plot_bin_label='Reconstructed Momentum', plot_binning_features=reconstructed_momentum, **kwargs)

def plot_energy_binned_performance(energy_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Energy', plot_binning_features=energy_features, **kwargs)

def plot_to_wall_binned_performance(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='To Wall', plot_binning_features=to_wall_features, **kwargs)

def plot_zenith_binned_performance(zenith_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Zenith', plot_binning_features=zenith_features, **kwargs)

def plot_azimuth_binned_performance(azimuth_features, **kwargs):
    return plot_single_var_binned_performance(plot_bin_label='Azimuth', plot_binning_features=azimuth_features, **kwargs)

# ========================================================================
# Multiple Variable Plotting Functions


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()


    
    #print(handles)
    #print(labels)

    #l = ax.legend([(p1,p2)],['points'],scatterpoints=2)
    handle_dict = dict((k, []) for k in labels)
    for handle, label in zip(handles, labels):
        handle_dict[label].append(handle)

    new_handles = [tuple(h) for h in handle_dict.values()]

    ax.legend(new_handles, labels, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, prop={'size': 16}, bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_multi_var_binned_performance(scores_list, labels_list, reconstructed_momentum, 
                            binning_features, binning_bin_size,
                            plot_binning_features, 
                            fpr_fixed_point, index_dict, recons_mom_bin_size=50, plot_bins=None, 
                            axes=None,marker='o--',colors=None,title_note='',metric='efficiency',yrange=None, xrange=None, names=None,
                            plot_bin_label=None):
    
    # TODO: fix cmap selection
    cmaps = ['Reds', 'Greens', 'Purples', 'Blues', 'Oranges',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    if axes is None:
        fig, axes = plt.subplots(1,1,figsize=(9,5), facecolor='w')

    for idx, _ in enumerate(scores_list):
        compute_multi_var_fixed_operating_performance(scores           = scores_list[idx], 
                                                labels                 = labels_list[idx],
                                                fixed_binning_features = reconstructed_momentum, 
                                                fixed_bin_size         = 50,
                                                binning_features       = binning_features,
                                                binning_bin_size       = binning_bin_size,
                                                plot_binning_features  = plot_binning_features, 
                                                plot_bins              = plot_bins,
                                                index_dict             = index_dict, 
                                                ignore_dict            = index_dict, 
                                                threshold              = 2, 
                                                ax                     = axes,
                                                muon_comparison        = False, 
                                                use_rejection          = False, 
                                                linecolor              = 'b',
                                                line_title             = 'No OD veto',
                                                metric                 = 'efficiency',
                                                cmap                   = plt.cm.get_cmap(cmaps[idx]))
    legend_without_duplicate_labels(axes)
    
    #axes.legend(names)

    

def plot_to_wall_binned_in_energy(to_wall_features, **kwargs):
    return plot_multi_var_binned_performance(binning_bin_label='To Wall', plot_bin_label='To Wall', fixed_binning_features=to_wall_features, **kwargs)

def plot_zenith_binned_in_azimuth(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(binning_bin_label='To Wall', fixed_binning_features=to_wall_features, **kwargs)

def plot_azimuth_binned_in_zenith(to_wall_features, **kwargs):
    return plot_single_var_binned_performance(binning_bin_label='To Wall', fixed_binning_features=to_wall_features, **kwargs)