"""
Utils for analyzing performance depending on some variable
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import stable_cumsum
# from sklearn.metrics import det_curve

from textwrap import wrap

from WatChMaL.analysis.multi_plot_utils import multi_compute_roc, multi_plot_roc
from WatChMaL.analysis.comparison_utils import multi_collapse_test_output
from WatChMaL.analysis.plot_utils import separate_particles


# ========================================================================
# Helper Functions
# TODO: move label from global to param
label_size = 14

def remove_indices(array, cut_idxs):
    return np.delete(array, cut_idxs, 0)

def get_filtered_particle_data(scores, labels, plot_binning_features, fixed_binning_features, index_dict, desired_labels):
    scores, labels, plot_binning_features, fixed_binning_features = separate_particles([scores, labels, plot_binning_features, fixed_binning_features],labels,index_dict, desired_labels=desired_labels) #,desired_labels=['$e$','$\mu$'])
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(plot_binning_features), np.concatenate(fixed_binning_features)


def compute_metrics(scores, labels, plotting_bin_idxs_list, true_label, false_label, thresholds_per_event, index_dict, metric='efficiency', efficiency_correction_factor=1., rejection_correction_factor=1., verbose=False):
    
    # Find metrics for each plot_binning_features bin
    bin_metrics, y_err = [],[]
    for bin_idxs in plotting_bin_idxs_list:
        pred_pos_idxs = np.where( scores[bin_idxs] >= thresholds_per_event[bin_idxs])[0]
        pred_neg_idxs = np.where( scores[bin_idxs] < thresholds_per_event[bin_idxs] )[0]

        fp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict[false_label])[0].shape[0]
        tp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict[true_label] )[0].shape[0]
        fn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict[true_label] )[0].shape[0]
        tn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict[false_label])[0].shape[0]
        
        if verbose:
            print(str(fp) + ' | ' + str(tp) + ' | ' + str(fn) + ' | ' + str(tn))
        
        # TODO: division by zero problem
        if metric == 'efficiency':
            performance = tp/(tp + fn + 1e-10)
            performance *= efficiency_correction_factor
        else:
            performance = fp/(fp + tn + 1e-10)
            performance *= rejection_correction_factor
        
        #N = len(bin_idxs) + 1e-10
        N = len(labels[bin_idxs[labels[bin_idxs] == index_dict[true_label]]]) + 1e-10

        bin_metrics.append(performance)
        y_err.append( np.sqrt(performance*(1 - performance) / N))

    return bin_metrics, y_err


def get_fixed_bin_assignments(fixed_binning_features, fixed_bin_size):
    '''
    Bin by fixed_binning_features
    '''
    fixed_bins_true = [0. + fixed_bin_size * i for i in range(math.ceil(np.max(fixed_binning_features)/fixed_bin_size))]   
    fixed_bins = fixed_bins_true[0:-1]

    recons_mom_bin_assignments = np.digitize(fixed_binning_features, fixed_bins)
    recons_mom_bin_idxs_list = [[]]*len(fixed_bins)

    for bin_idx in range(len(fixed_bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments == bin_num)[0]
    
    return fixed_bins_true, recons_mom_bin_idxs_list

def get_binning_bin_assignments(binning_bin_features, binning_bin_size):
    '''
    Bin by binning_bin_features
    '''
    fixed_bins_true = [0. + binning_bin_size * i for i in range(math.ceil(np.max(binning_bin_features)/binning_bin_size))]   
    fixed_bins = fixed_bins_true[0:-1]

    recons_mom_bin_assignments = np.digitize(binning_bin_features, fixed_bins)
    recons_mom_bin_idxs_list = [[]]*len(fixed_bins)

    for bin_idx in range(len(fixed_bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments == bin_num)[0]
    
    return fixed_bins_true, recons_mom_bin_idxs_list


def get_plot_bin_assignments(plot_binning_features, plot_bins):
    '''
    Bin by plot_binning_features
    '''
    if isinstance(plot_bins, int):
        _, true_bins = np.histogram(plot_binning_features, bins=plot_bins, range=(np.min(plot_binning_features), np.max(plot_binning_features)))
    else:        
        true_bins = plot_bins
    
    bins = true_bins[0:-1]

    true_mom_bin_assignments = np.digitize(plot_binning_features, bins)
    plot_bin_idxs_list = [[]]*len(bins)

    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        plot_bin_idxs_list[bin_idx] = np.where(true_mom_bin_assignments == bin_num)[0]
    
    return true_bins, plot_bin_idxs_list


def get_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, index_dict,  efficiency_correction_factor, rejection_correction_factor):
    '''
    Compute thresholds giving fixed fpr per fixed_binning_features bin
    '''
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list): 
        # TODO: include bin only if shape > 0
        if bin_idxs.shape[0] > 0:
            fps, tps, thresholds = binary_clf_curve(labels[bin_idxs], scores[bin_idxs], pos_label=index_dict['$e$'])

            fns = tps[-1] - tps
            tns = fps[-1] - fps
            fprs = fps/(fps + tns)

            fprs *= rejection_correction_factor

            operating_point_idx = (np.abs(fprs - fpr_fixed_point)).argmin()
            thresholds_per_event[bin_idxs] = thresholds[operating_point_idx]
        else:
            print("Empty bin")
    
    return thresholds_per_event


def get_pion_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, p0, p1, pi0mass, index_dict):
    '''
    Compute thresholds giving fixed fpr per fixed_binning_features bin
    '''
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list): 
        # TODO: include bin only if shape > 0
        if bin_idxs.shape[0] > 0:
            if bin_idx > len(p0) - 1:
                thresholds_per_event[bin_idxs] = -(p0[-1] + p1[-1]*pi0mass[bin_idxs])
            else:
                thresholds_per_event[bin_idxs] = -(p0[bin_idx] + p1[bin_idx]*pi0mass[bin_idxs])

        else:
            print("Empty bin")
    
    return thresholds_per_event


# ========================================================================
# Single Variable Plot Functions

def compute_fixed_operating_performance(scores, labels, fixed_binning_features, plot_binning_features, fpr_fixed_point, index_dict, fixed_bin_size=50, plot_bins=20, metric='efficiency', desired_labels=['$e$','$\mu$'], efficiency_correction_factor=1., rejection_correction_factor=1., verbose=False):
    '''
    Plots a metric as a function of a physical parameter, at a fixed operating point of another metric.

    plot_fitqun_binned_performance(scores, labels, true_momentum, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, true_mom_bins=20, 
                            ax=None,marker='o',color='k',title_note='',metric='efficiency',yrange=None)
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

    assert plot_binning_features.shape[0]  == scores.shape[0], 'Error: plot_binning_features must have same length as softmaxes'
    assert fixed_binning_features.shape[0] == scores.shape[0], 'Error: fixed_binning_features must have same length as softmaxes'
    assert len(desired_labels) == 2, 'Error: must have a single true and single negative label'
    
    # Remove gamma events
    true_label = desired_labels[0]
    false_label = desired_labels[1]

    scores, labels, plot_binning_features, fixed_binning_features = get_filtered_particle_data(scores, labels, plot_binning_features, fixed_binning_features, index_dict, desired_labels)
    
    # Bin by fixed_binning_features
    _, recons_mom_bin_idxs_list = get_fixed_bin_assignments(fixed_binning_features, fixed_bin_size)

    # Bin by plot_binning_features
    thresholds_per_event = get_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, index_dict, efficiency_correction_factor, rejection_correction_factor)

    true_bins, plot_bin_idxs_list = get_plot_bin_assignments(plot_binning_features, plot_bins)

    # Find metrics for each plot_binning_features bin
    bin_metrics, yerr = compute_metrics(scores = scores, 
                                        labels = labels,
                                        plotting_bin_idxs_list = plot_bin_idxs_list,
                                        true_label = true_label, 
                                        false_label = false_label,
                                        thresholds_per_event = thresholds_per_event,
                                        index_dict = index_dict,
                                        metric = metric,
                                        efficiency_correction_factor=efficiency_correction_factor,
                                        rejection_correction_factor=rejection_correction_factor)

    # Compute bin centers
    bin_centers = (true_bins[:-1] + true_bins[1:]) / 2

    return bin_centers, bin_metrics, yerr


def compute_pion_fixed_operating_performance(
                                     scores, labels, 
                                     fixed_binning_features,  
                                     plot_binning_features, 
                                     p0, p1, pi0mass,
                                     fpr_fixed_point, index_dict, fixed_bin_size=50, plot_bins=20, metric='efficiency'):
    '''
    Plots a metric as a function of a physical parameter, at a fixed operating point of another metric using pion (2 parameter) cuts.
    TODO: scores must be fq1rnll[0][1] - fqpi0nll[0]
    '''

    assert plot_binning_features.shape[0]  == scores.shape[0], 'Error: plot_binning_features must have same length as softmaxes'
    assert fixed_binning_features.shape[0] == scores.shape[0], 'Error: fixed_binning_features must have same length as softmaxes'
    
    true_label  = '$e$'
    false_label = '$\pi^0$'

    # Remove gamma events
    # TODO: remember pion label dict
    desired_labels=['$e$','$\pi^0$']
    scores, labels, plot_binning_features, fixed_binning_features = get_filtered_particle_data(scores, labels, plot_binning_features, fixed_binning_features, index_dict, desired_labels)

    # Bin by fixed_binning_features
    _, recons_mom_bin_idxs_list = get_fixed_bin_assignments(fixed_binning_features, fixed_bin_size)

    # Compute thresholds giving fixed fpr per fixed_binning_features bin
    thresholds_per_event = get_pion_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, p0, p1, pi0mass, index_dict)
    
    # Bin by plot_binning_features
    true_bins, plot_bin_idxs_list = get_plot_bin_assignments(plot_binning_features, plot_bins)
    
    # Find metrics for each plot_binning_features bin
    bin_metrics, y_err = compute_metrics(scores = scores, 
                                         labels = labels,
                                         plotting_bin_idxs_list = plot_bin_idxs_list,
                                         true_label = true_label, 
                                         false_label = false_label,
                                         thresholds_per_event = thresholds_per_event,
                                         index_dict = index_dict,
                                         metric = metric)

    bin_centers = (true_bins[:-1] + true_bins[1:]) / 2
    
    return bin_centers, bin_metrics, y_err


def plot_fixed_operating_performance(bin_centers, bin_metrics, yerr, marker, color, name, fixed_bin_label, plot_bin_label, plot_bin_units,
                                     fpr_fixed_point, title_note, metric=None, yrange=None, xrange=None, ax=None, publication_style=True, show_x_err=False, desired_labels=['$e$','$\mu$'], show_legend=True):

    true_label  = desired_labels[0]
    false_label = desired_labels[1]

    if metric == 'efficiency':
        # '$e$- Efficiency of $\pi_0$ Rejection Cut'
        metric_name = '{} Signal Efficiency'.format(true_label) #'$e$- Signal Efficiency'
    else :
        metric_name = '{} Mis-ID Rate'.format(false_label) #'$\mu$- Mis-ID Rate'
    
    title = '{} vs {} At {} Fixed Bin {} Mis-ID Rate'.format(metric_name, plot_bin_label, fixed_bin_label, false_label) 
    if title_note == 'fpr_fixed_point':
        title = title + ' {}%'.format(fpr_fixed_point*100)
    else:
        title = title + '{}'.format(title_note)
    title = "\n".join(wrap(title, 60))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))

    if show_x_err:
        x_err = (bin_centers[1] - bin_centers[0])/2*np.ones_like(yerr)
    else:
        x_err = None

    if publication_style:
        name = name + ', {} Mis-ID Rate {:.2f}%'.format(false_label, fpr_fixed_point*100)

    if not show_legend:
        name = None
    
    ax.errorbar(bin_centers, bin_metrics, yerr=yerr, xerr=x_err, fmt=marker, color=color, label=name, ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
    
    if not publication_style:
        ax.grid(b=True, which='major', color='gray', linestyle='--')

    ax.set_ylabel(metric_name, fontsize=label_size)
    ax.set_xlabel("{} [{}]".format(plot_bin_label, plot_bin_units), fontsize=label_size)
    ax.set_title(title, fontsize=1.1*label_size)

    if yrange is not None: 
        ax.set_ylim(yrange) 
    
    if xrange is not None: 
        ax.set_xlim(xrange) 
    
    secax = ax.secondary_yaxis('right')

    ax.legend(handlelength=5.0 if not (marker == 'o') else None)#, prop={'size': label_size})

# ========================================================================
# Multiple Variable Plot Functions

def compute_multi_var_fixed_operating_performance(
                             scores, labels,
                             fixed_binning_features, fixed_bin_size,
                             binning_features, binning_bin_size, 
                             plot_binning_features, plot_bins,
                             index_dict, ignore_dict, fpr_fixed_point, axes1=None, axes2=None, 
                             linecolor='b', line_title=None,
                             metric='efficiency', ax=None, cmap=None, desired_labels=['$e$','$\mu$'], efficiency_correction_factor=1., rejection_correction_factor=1.,):
    '''
    Plot performance as a function of a single variable
    '''

    assert plot_binning_features.shape[0]  == scores.shape[0], 'Error: plot_binning_features must have same length as softmaxes'
    assert binning_features.shape[0]  == scores.shape[0], 'Error: binning_features must have same length as softmaxes'
    assert fixed_binning_features.shape[0] == scores.shape[0], 'Error: fixed_binning_features must have same length as softmaxes'

    # Remove gamma events
    # TODO: make desired labels dynamic
    true_label  = desired_labels[0]
    false_label = desired_labels[1]

    scores, labels, plot_binning_features, fixed_binning_features, binning_features = separate_particles([scores, labels, plot_binning_features, fixed_binning_features, binning_features],labels,index_dict, desired_labels=desired_labels) #,desired_labels=['$e$','$\mu$'])
    scores, labels, plot_binning_features, fixed_binning_features, binning_features = np.concatenate(scores), np.concatenate(labels), np.concatenate(plot_binning_features), np.concatenate(fixed_binning_features), np.concatenate(binning_features)

    get_filtered_particle_data(scores, labels, plot_binning_features, fixed_binning_features, index_dict, desired_labels)
    
    ####### Bin by fixed_binning_features #######
    _, recons_mom_bin_idxs_list = get_fixed_bin_assignments(fixed_binning_features, fixed_bin_size)

    # Compute thresholds giving fixed fpr per fixed_binning_features bin
    thresholds_per_event = get_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, index_dict, efficiency_correction_factor, rejection_correction_factor)
    
    ####### Bin by binning_features #######
    # TODO: fix to allow either bin size or number
    #true_bins, binning_bin_idxs_list = get_binning_bin_assignments(binning_features, binning_bin_size)
    _, true_bins = np.histogram(binning_features, bins=plot_bins, range=(np.min(binning_features), np.max(binning_features)))
    binning_bins = true_bins[0:-1]

    recons_mom_bin_assignments = np.digitize(binning_features, binning_bins)
    binning_bin_idxs_list = [[]]*len(true_bins)

    for bin_idx in range(len(true_bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        binning_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments == bin_num)[0]

    ####### Bin by plot_binning_features #######
    if isinstance(plot_bins, int):
        _, true_plotting_bins = np.histogram(plot_binning_features, bins=plot_bins, range=(np.min(plot_binning_features), np.max(plot_binning_features)))
    else:        
        true_plotting_bins = plot_bins
    
    plotting_bins = true_plotting_bins[0:-1]

    plotting_bin_assignments = np.digitize(plot_binning_features, plotting_bins)

    # For each binning_features bin plot in other features with fixed momentum
    # TODO: restore
    all_true_plotting_bins, all_bin_metrics, all_yerr = [],[],[]
    for idx, bin_idxs in enumerate(binning_bin_idxs_list):
        subset_scores = scores[bin_idxs]
        subset_labels = labels[bin_idxs]

        subset_plot_binning_features = plotting_bin_assignments[bin_idxs]
        subset_thresholds_per_event  = thresholds_per_event[bin_idxs]

        '''
        _, subset_plot_bin_idxs_list = get_plot_bin_assignments(plotting_bin_assignments[bin_idxs], plot_bins)
        '''

        
        subset_plot_bin_idxs_list = [[]]*len(plotting_bins)

        for bin_idx in range(len(plotting_bins)):
            bin_num = bin_idx + 1 #these are one-indexed for some reason
            subset_plot_bin_idxs_list[bin_idx] = np.where(subset_plot_binning_features == bin_num)[0]

        # TODO: verify correctness
        bin_metrics, y_err = compute_metrics(scores = subset_scores, 
                                             labels = subset_labels,
                                             plotting_bin_idxs_list = subset_plot_bin_idxs_list,
                                             thresholds_per_event = subset_thresholds_per_event,
                                             true_label = true_label, 
                                             false_label = false_label,
                                             index_dict = index_dict,
                                             metric = metric,
                                             efficiency_correction_factor = efficiency_correction_factor, 
                                             rejection_correction_factor = rejection_correction_factor)

        all_true_plotting_bins.append(true_plotting_bins)
        all_bin_metrics.append(bin_metrics)
        all_yerr.append(y_err)

    return all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins



def compute_pion_multi_var_fixed_operating_performance(
                             scores, labels,
                             fixed_binning_features, fixed_bin_size,
                             binning_features, binning_bin_size, 
                             plot_binning_features, plot_bins,
                             p0, p1, pi0mass,
                             index_dict, ignore_dict, fpr_fixed_point, axes1=None, axes2=None, 
                             linecolor='b', line_title=None,
                             metric='efficiency', ax=None, cmap=None, desired_labels=['$e$','$\mu$']):
    '''
    Plot performance as a function of a single variable
    '''

    assert plot_binning_features.shape[0]  == scores.shape[0], 'Error: plot_binning_features must have same length as softmaxes'
    assert binning_features.shape[0]  == scores.shape[0], 'Error: binning_features must have same length as softmaxes'
    assert fixed_binning_features.shape[0] == scores.shape[0], 'Error: fixed_binning_features must have same length as softmaxes'

    # Remove gamma events
    # TODO: make desired labels dynamic
    true_label  = desired_labels[0]
    false_label = desired_labels[1]

    scores, labels, plot_binning_features, fixed_binning_features, binning_features = separate_particles([scores, labels, plot_binning_features, fixed_binning_features, binning_features],labels,index_dict, desired_labels=desired_labels) #,desired_labels=['$e$','$\mu$'])
    scores, labels, plot_binning_features, fixed_binning_features, binning_features = np.concatenate(scores), np.concatenate(labels), np.concatenate(plot_binning_features), np.concatenate(fixed_binning_features), np.concatenate(binning_features)

    get_filtered_particle_data(scores, labels, plot_binning_features, fixed_binning_features, index_dict, desired_labels)
    
    ####### Bin by fixed_binning_features #######
    _, recons_mom_bin_idxs_list = get_fixed_bin_assignments(fixed_binning_features, fixed_bin_size)

    # Compute thresholds giving fixed fpr per fixed_binning_features bin
    thresholds_per_event = get_pion_threshold_assignments(scores, labels, recons_mom_bin_idxs_list, fpr_fixed_point, p0, p1, pi0mass, index_dict)
    
    ####### Bin by binning_features #######
    true_bins, binning_bin_idxs_list = get_binning_bin_assignments(binning_features, binning_bin_size)

    ####### Bin by plot_binning_features #######
    if isinstance(plot_bins, int):
        _, true_plotting_bins = np.histogram(plot_binning_features, bins=plot_bins, range=(np.min(plot_binning_features), np.max(plot_binning_features)))
    else:        
        true_plotting_bins = plot_bins
    
    plotting_bins = true_plotting_bins[0:-1]

    plotting_bin_assignments = np.digitize(plot_binning_features, plotting_bins)
    

    # For each binning_features bin plot in other features with fixed momentum
    # TODO: restore
    all_true_plotting_bins, all_bin_metrics, all_yerr = [],[],[]
    for idx, bin_idxs in enumerate(binning_bin_idxs_list):
        subset_scores = scores[bin_idxs]
        subset_labels = labels[bin_idxs]

        subset_plot_binning_features = plotting_bin_assignments[bin_idxs]
        subset_thresholds_per_event  = thresholds_per_event[bin_idxs]

        '''
        _, subset_plot_bin_idxs_list = get_plot_bin_assignments(plotting_bin_assignments[bin_idxs], plot_bins)
        '''

        
        subset_plot_bin_idxs_list = [[]]*len(plotting_bins)

        for bin_idx in range(len(plotting_bins)):
            bin_num = bin_idx + 1 #these are one-indexed for some reason
            subset_plot_bin_idxs_list[bin_idx] = np.where(subset_plot_binning_features == bin_num)[0]
        

        # TODO: verify correctness
        bin_metrics, y_err = compute_metrics(scores = subset_scores, 
                                             labels = subset_labels,
                                             plotting_bin_idxs_list = subset_plot_bin_idxs_list,
                                             thresholds_per_event = subset_thresholds_per_event,
                                             true_label = true_label, 
                                             false_label = false_label,
                                             index_dict = index_dict,
                                             metric = metric)

        all_true_plotting_bins.append(true_plotting_bins)
        all_bin_metrics.append(bin_metrics)
        all_yerr.append(y_err)

    return all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins



def plot_multi_var_fixed_operating_performance(all_true_plotting_bins, all_bin_metrics, all_yerr, true_bins, marker, cmap, 
                                               fixed_bin_label, binning_bin_label, plot_bin_label, 
                                               fixed_bin_units, binning_bin_units, plot_bin_units,
                                               fpr_fixed_point, title_note, metric=None, yrange=None, xrange=None, ax=None, desired_labels=['$e$','$\mu$']):

    # TODO: make desired labels dynamic
    true_label  = desired_labels[0]
    false_label = desired_labels[1]

    c = cmap(np.linspace(0.4,1,len(all_true_plotting_bins)))
    
    # Plot metrics
    if metric == 'efficiency':
        metric_name = '{} Signal Efficiency'.format(true_label)
    else:
        metric_name ='{} Mis-ID Rate'.format(false_label)
    
    title = '{} vs {} At {} Bin {} Mis-ID Rate of {}%{}'.format(metric_name, plot_bin_label, fixed_bin_label, false_label, fpr_fixed_point*100, title_note)
    title = "\n".join(wrap(title, 60))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6), facecolor='w')
    #for idx, true_plotting_bins in enumerate(all_true_plotting_bins):
    #    print(idx)

    for idx, true_plotting_bins in enumerate(all_true_plotting_bins):
        bin_centers = (true_plotting_bins[:-1] + true_plotting_bins[1:]) / 2
        #print(idx)
        #print(len(all_true_plotting_bins))
        plot_label = '{} Range ${:.1f}-{:.1f}$ {}'.format(binning_bin_label, true_bins[idx], true_bins[idx + 1], binning_bin_units)
        ax.errorbar(bin_centers, all_bin_metrics[idx], yerr=all_yerr[idx],fmt=marker, color=c[idx], ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2, label=plot_label)
        ax.grid(b=True, which='major', color='gray', linestyle='--')

        ax.set_ylabel(metric_name)
        ax.set_xlabel("{}  [{}]".format(plot_bin_label, plot_bin_units), fontsize=label_size)
        ax.set_title(title)

        if yrange is not None: 
            ax.set_ylim(yrange) 
        
        if xrange is not None: 
            ax.set_xlim(xrange) 
        
        secax = ax.secondary_yaxis('right')
            
        ax.legend(prop={'size': 16}, bbox_to_anchor=(1.05, 1), handlelength=5.0, loc='upper left')


def compute_2D_hist():
    pass


# ========================================================================
# Pion Plot Functions (needed for 2 parameter cuts)

def plot_binned_response(softmaxes, labels, particle_names, binning_features, binning_label,efficiency, bins, p_bins, index_dict, extra_panes=None, log_scales=[], legend_label_dict=None, wrap_size=35):
    '''
    Plot softmax response, binned in a feature of the event.
    Args:
        softmaxes                   ... 2d array of softmax output, shape (nsamples, noutputs)
        labels                      ... 1d array of labels, length n_samples
        particle_names              ... string particle names for which to plot the response, must be keys of index_dict
        binning_features            ... 1d array of feature to use in binning, length n_samples
        binning_label               ... string, name of binning feature to use in title and x-axis label
        efficiency                  ... bin signal efficiency to fix
        bins                        ... number of bins to use in feature histogram
        p_bins                      ... number of bins to use in probability density histogram
        index_dict                  ... dictionary of particle labels, must have all of particle_names as keys, pointing to values taken by 'labels'
        extra_panes                 ... list of lists of particle names to combine into an "extra output" e.g. [["e", "gamma"]] adds the P(e-)+P(gamma) pane
        log_scales                  ... indices of axes.flatten() to which to apply log color scaling
        legend_label_dict           ... dictionary of display symbols for each string label, to use for displaying pretty characters
        wrap_size                   ... width of box to wrap title into
    author: Calum Macdonald
    June 2020
    '''

    if legend_label_dict is None:
        legend_label_dict = {}
        for name in particle_names: legend_label_dict[name] = name

    noutputs = softmaxes.shape[1]
    if extra_panes is None: extra_panes=[]

    label_size = 18
    fig, axes = plt.subplots(len(particle_names),noutputs+len(extra_panes),figsize=(12*len(particle_names),12*noutputs+len(extra_panes)), facecolor='w')

    #axes = axes.reshape(-1)
    log_axes = axes.flatten()[log_scales]

    #bin by whatever feature
    if isinstance(bins, int):
        _,bins = np.histogram(binning_features, bins=bins)
    b_bin_centers = [bins[i] + (bins[i+1]-bins[i])/2 for i in range(bins.shape[0]-1)]
    binning_edges=bins
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)
    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments==bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs]})
    
    edges = None
    for output_idx in range(noutputs):
        for particle_idx, particle_name in [(index_dict[particle_name], particle_name) for particle_name in particle_names]:
            ax = axes[particle_idx][output_idx]
            data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
            means = []
            stddevs = []
            for bin_idx, bin in enumerate(bin_data):
                relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict, desired_labels=[particle_name])[0][0]

                if edges is None: ns, edges = np.histogram(relevant_softmaxes[:,output_idx], bins=p_bins,density=True,range=(0.,1.))
                else: ns, _ = np.histogram(relevant_softmaxes[:,output_idx], bins=edges,density=True)

                data[bin_idx, :] = ns
                p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                means.append(np.mean(relevant_softmaxes[:,output_idx]))
                stddevs.append(np.std(relevant_softmaxes[:,output_idx]))                

            if ax in log_axes:
                min_value = np.unique(data)[1]
                data = np.where(data==0, min_value, data)
            mesh = ax.pcolormesh(binning_edges, edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
            fig.colorbar(mesh,ax=ax)
            ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
            ax.set_xlabel(binning_label,fontsize=label_size)
            ax.set_ylabel('P({})'.format(legend_label_dict[particle_names[output_idx]]),fontsize=label_size)
            ax.set_ylim([0,1])
            ax.set_title('\n'.join(wrap('P({}) Density For {} Events vs {}'.format(legend_label_dict[particle_names[output_idx]],legend_label_dict[particle_name],binning_label),wrap_size)),fontsize=label_size)

    for n, extra_pane_particle_names in enumerate(extra_panes):
        for particle_idx, particle_name in [(index_dict[particle_name], particle_name) for particle_name in particle_names]:
                ax = axes[particle_idx][n+noutputs]
                data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
                means = []
                stddevs = []
                for bin_idx, bin in enumerate(bin_data):
                    relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict, desired_labels=[particle_name])[0][0]
                    extra_output = reduce(lambda x,y : x+y, [relevant_softmaxes[:,index_dict[pname]] for pname in extra_pane_particle_names])
                    ns, _ = np.histogram(extra_output, bins=edges,density=True)
                    data[bin_idx, :] = ns
                    p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                    means.append(np.mean(extra_output))
                    stddevs.append(np.std(extra_output))
                if ax in log_axes:
                    min_value = np.unique(data)[1]
                    data = np.where(data==0, min_value, data)
                mesh = ax.pcolormesh(binning_edges,edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
                fig.colorbar(mesh,ax=ax)
                ax.set_ylim([0,1])
                ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
                ax.set_xlabel(binning_label,fontsize=label_size)
                extra_output_label = reduce(lambda x,y: x + ' + ' + f"P({y})", [f'P({legend_label_dict[name]})' if i==0 else legend_label_dict[name] for i, name in enumerate(extra_pane_particle_names)])
                ax.set_ylabel(extra_output_label,fontsize=label_size)
                ax.set_title('\n'.join(wrap('{} Density For {} Events vs {}'.format(extra_output_label, legend_label_dict[particle_name],binning_label),wrap_size)),fontsize=label_size)


def plot_2d_hist_ratio(dist_1_x,dist_1_y,dist_2_x, dist_2_y,bins=(150,150),fig=None,ax=None,
                  title=None, xlabel=None, ylabel=None, ratio_range=None):
    '''
    Plots the 2d ratio between the 2d histograms of two distributions.
    Args:
        dist_1_x:               ... 1d array of x-values of distribution 1 of length n
        dist_1_y:               ... 1d array of y-values of distribution 1 of length n
        dist_2_x:               ... 1d array of x-values of distribution 2 of length n
        dist_2_y:               ... 1d array of y-values of distribution 2 of length n
        bins:                   ... tuple of integer numbers of bins in x and y 
        fig                     ... figure on which to plot
        ax                      ... axis on which to plot ratio
        title                   ... str, title
        xlabel                  ... str, x-axis label
        ylabel                  ... str, y-axis label
        ratio_range             ... range of ratios to display
    author: Calum Macdonald
    May 2020
    '''
    if ax is None: 
        fig,ax = plt.subplots(1,1,figsize=(8,8))
    
    bin_range = [[np.min([np.min(dist_1_x),np.min(dist_2_x)]),np.max([np.max(dist_1_x),np.max(dist_2_x)])],
             [np.min([np.min(dist_1_y),np.min(dist_2_y)]),np.max([np.max(dist_1_y),np.max(dist_2_y)])]]
    
    ns_1, xedges, yedges = np.histogram2d(dist_1_x,dist_1_y,bins=bins,density=True,range=bin_range)
    ns_2,_,_ = np.histogram2d(dist_2_x,dist_2_y,bins=bins,density=True,range=bin_range)
    
    ratio = ns_1/ns_2
    ratio = np.where((ns_2==0) & (ns_1==0),1,ratio)
    ratio = np.where((ns_2==0) & (ns_1!=0),10,ratio)
    
    pc = ax.pcolormesh(xedges, yedges, np.swapaxes(ratio,0,1),vmin=ratio_range[0],vmax=ratio_range[1],cmap="RdBu_r")
    fig.colorbar(pc, ax=ax)

    if title is not None: 
        ax.set_title(title)
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)
    
    return fig


def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    '''
    ###########################################################################################
    TODO: This functionality added as det_curve in sklearn v0.24, switch when container updated
    ###########################################################################################

    SOURCE: Scikit.metrics internal usage tool
    Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    '''

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]

    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]
