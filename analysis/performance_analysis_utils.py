"""
Utils for analyzing performance depending on some variable
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import stable_cumsum
# from sklearn.metrics import det_curve

from WatChMaL.analysis.multi_plot_utils import multi_compute_roc, multi_plot_roc
from WatChMaL.analysis.comparison_utils import multi_collapse_test_output
from WatChMaL.analysis.plot_utils import separate_particles

def get_idxs_satisfying_bounds(data_values, lower_bound, upper_bound):
    return np.where((lower_bound < data_values) & (data_values <= upper_bound))[0]


def plot_binned_performance(softmaxes, labels, binning_features, binning_label, bins, fixed='rejection', operating_point, metric='purity', index_dict, 
                            label_0, label_1, ax=None,marker='o--', color=None, legend_label_dict=None, title_note=''):
    '''
    Plots a performance metric as a function of a physical parameter, at a fixed operating point of another metric.

    Args:
        softmaxes                      ... 2d array with first dimension n_samples
        labels                         ... 1d array of true event labels
        binning_features               ... 1d array of features for generating bins, eg. energy
        binning_label                  ... name of binning feature, to be used in title and xlabel
        efficiency                     ... signal efficiency per bin, to be fixed
        bins                           ... either an integer (number of evenly spaced bins) or list of n_bins+1 edges
        index_dict                     ... dictionary of particle string keys and values corresponding to labels in 'labels'
        label_0                        ... string, positive particle label, must be key of index_dict
        label_1                        ... string, negative particle label, must be key of index_dict
        metric                         ... string, metric to plot ('purity' for signal purity, 'rejection' for rejection fraction, 'efficiency' for signal efficiency)
        ax                             ... axis on which to plot
        color                          ... marker color
        marker                         ... marker type
        legend_label_dict              ... dictionary of display symbols for each string label, to use for displaying pretty characters
        title_note                     ... string to append to the plot title
    author: Calum Macdonald
    June 2020
    '''
    if legend_label_dict is None:
            legend_label_dict={}
            for name in [label_0, label_1]:
                legend_label_dict[name] = name
    
    label_size = 14

    assert binning_features.shape[0] == softmaxes.shape[0], 'Error: binning_features must have same length as softmaxes'
    
    # bin by given features
    if isinstance(bins, int):
        _, bins = np.histogram(binning_features, bins=bins)
    
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)

    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments == bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs], 'n' : this_bin_idxs.shape[0]})

    # compute efficiency, thresholds, purity per bin
    bin_metrics = []
    for bin_idx, data in enumerate(bin_data):
        (softmaxes_0,softmaxes_1), (labels_0,labels_1) = separate_particles([data['softmaxes'],data['labels']],data['labels'],index_dict,desired_labels=[label_0,label_1])

        fps, tps, thresholds = binary_clf_curve(np.concatenate((labels_0,labels_1)),np.concatenate((softmaxes_0,softmaxes_1))[:,index_dict[label_0]], pos_label=index_dict[label_0])
        
        # TODO: verify this step
        fns = tps[-1] - tps
        tns = fps[-1] - fps

        efficiencies = tps/(tps + fns)

        if fixed == 'efficiency':
            fixed_metric = efficiencies
        elif fixed == 'rejection':
            fixed_metric = tns / (tns + fps)
        elif fixed == 'purity': 
            performance = tps/(tps + fps)
        elif fixed == 'inverse fpr': 
            performance = np.where(fps != 0, (fps + tns) / fps, fps + tns)

        operating_point_idx = (np.abs(fixed_metric - operating_point)).argmin()

        if metric == 'efficiency':
            performance = efficiencies
        elif metric == 'rejection': 
            performance = tns / (tns + fps)
        elif metric == 'purity': 
            performance = tps/(tps + fps)
        elif metric == 'inverse fpr': 
            performance = np.where(fps != 0, (fps + tns) / fps, fps + tns)
        
        bin_metrics.append((fixed_metric[operating_point_idx], performance[operating_point_idx], np.sqrt(tns[operating_point_idx])/(tns[operating_point_idx] + fps[operating_point_idx])))
    
    bin_metrics = np.array(bin_metrics)

    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(binning_features) - bins[-1])/2 + bins[-1])

    if metric == 'purity':
        metric_name = '{}-{} Signal Purity'.format(label_0,label_1) 
    elif metric=='rejection': 
        metric_name =  '{} Rejection Fraction'.format(legend_label_dict[label_1]) 
    else: 
        metric_name = '{} Rejection'.format(legend_label_dict[label_1])

    # TODO: fix titling
    title = '{} \n vs {} At Bin {} Signal Efficiency {}{}'.format(metric_name, binning_label, legend_label_dict[label_0], operating_point, title_note)
    
    # plot results
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))
    
    ax.errorbar(bin_centers,bin_metrics[:,1],yerr=bin_metrics[:,2],fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
    ax.set_ylabel(metric_name, fontsize=label_size)
    ax.set_xlabel(binning_label, fontsize=label_size)
    ax.set_title(title)
    
    if metric == 'inverse fpr': 
        ax.set_yscale('log')


def multi_plot_fixed_operating_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax_list, subset_labels_list,
                             label_dict, ignore_dict, threshold,muon_comparison=False, use_rejection=False, linecolors=None, line_titles=None):
    fig1, axes1 = plt.subplots(3, 3, figsize=(25, 25), facecolor='w')
    fig2, axes2 = plt.subplots(3, 3, figsize=(25*1.5, 25), facecolor='w')

    for idx, (subset_softmax, subset_labels) in enumerate(zip(subset_softmax_list, subset_labels_list)):
        # TODO: fix args
        plot_fixed_operating_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax, subset_labels, label_dict, ignore_dict, threshold, axes1, axes2, muon_comparison, use_rejection, 
                                         linecolors[idx], line_titles[idx])

def plot_fixed_operating_performance(scores, labels, fixed_binning_features, plot_binning_features, 
                                     fpr_fixed_point, index_dict, fixed_bin_size=50, plot_bins=20, 
                                     ax=None,marker='o--',color='k',title_note='',metric='efficiency',yrange=None):
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

    assert plot_binning_features.shape[0] == scores.shape[0], 'Error: plot_binning_features must have same length as softmaxes'
    assert fixed_binning_features.shape[0] == scores.shape[0], 'Error: fixed_binning_features must have same length as softmaxes'

    label_size = 14

    # remove gamma events
    scores, labels, plot_binning_features, fixed_binning_features = separate_particles([scores, labels, plot_binning_features, fixed_binning_features],labels,index_dict,desired_labels=['$e$','$\mu$'])
    scores, labels, plot_binning_features, fixed_binning_features = np.concatenate(scores), np.concatenate(labels), np.concatenate(plot_binning_features), np.concatenate(fixed_binning_features)

    # bin by fixed_binning_features
    fixed_bins = [0. + fixed_bin_size * i for i in range(math.ceil(np.max(fixed_binning_features)/fixed_bin_size))]   
    fixed_bins = bins[0:-1]

    recons_mom_bin_assignments = np.digitize(fixed_binning_features, fixed_bins)
    recons_mom_bin_idxs_list = [[]]*len(fixed_bins)

    for bin_idx in range(len(fixed_bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments == bin_num)[0]

    # compute threshold giving fixed fpr per fixed_binning_features bin
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list): 
        # TODO: include bin only if shape > 0
        if bin_idxs.shape[0] > 0:
            fps, tps, thresholds = binary_clf_curve(labels[bin_idxs], scores[bin_idxs], pos_label=index_dict['$e$'])

            fns = tps[-1] - tps
            tns = fps[-1] - fps
            fprs = fps/(fps + tns)

            operating_point_idx = (np.abs(fprs - fpr_fixed_point)).argmin()
            thresholds_per_event[bin_idxs] = thresholds[operating_point_idx]
        else:
            print("Empty bin")

    #bin by true momentum
    ns, true_bins = np.histogram(plot_binning_features, bins=plot_bins, range=(np.min(plot_binning_features), np.max(plot_binning_features))) #, range=(200., np.max(plot_binning_features)) if metric=='mu fpr' else (0,1000))
    bins = true_bins[0:-1]
    true_mom_bin_assignments = np.digitize(plot_binning_features, bins)
    true_mom_bin_idxs_list = [[]]*len(bins)

    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        true_mom_bin_idxs_list[bin_idx] = np.where(true_mom_bin_assignments == bin_num)[0]

    # find metrics for each true momentum bin
    bin_metrics=[]
    for bin_idxs in true_mom_bin_idxs_list:
        pred_pos_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] > 0)[0]
        pred_neg_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] < 0)[0]

        fp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['$\mu$'] )[0].shape[0]
        tp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['$e$'] )[0].shape[0]
        fn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['$e$'] )[0].shape[0]
        tn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['$\mu$'] )[0].shape[0]
        
        # TODO: division by zero
        if metric == 'efficiency':
            bin_metrics.append(tp/(tp+fn))
        else:
            bin_metrics.append(fp/(fp + tn))

    #plot metrics
    bin_centers =  (true_bins[:-1] + true_bins[1:]) / 2
    
     #[(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins) - 1)] #range(0,len(bins)-1)]
    #bin_centers.append((np.min(plot_binning_features), np.max(plot_binning_features))#(np.max(plot_binning_features) - bins[-1])/2 + bins[-1] if metric=='mu fpr' else (1000 - bins[-1])/2 + bins[-1])

    if metric == 'efficiency':
        metric_name = 'e- Signal Efficiency'
    else :
        metric_name ='\u03BC- Mis-ID Rate'
    title = '{} \n vs True Momentum At Reconstructed Momentum Bin \u03BC- Mis-ID Rate of {}%{}'.format(metric_name, fpr_fixed_point*100, title_note)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))
    
    # Compute errors
    yerr = np.ones_like(bin_metrics)
    yerr = np.zeros_like(bin_metrics)

    # plot
    ax.errorbar(bin_centers, bin_metrics, yerr=np.zeros_like(bin_metrics),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
    ax.grid(b=True, which='major', color='gray', linestyle='--')
    # nax = ax.twinx()
    ax.set_ylabel(metric_name)
    ax.set_xlabel("True Momentum (MeV/c)", fontsize=label_size)
    ax.set_title(title)

    if yrange is not None: 
        ax.set_ylim(yrange) 

    return plot_binning_features, thresholds_per_event

def plot_fitqun_binned_performance(scores, labels, true_momentum, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, true_mom_bins=20, 
                            ax=None,marker='o--',color='k',title_note='',metric='efficiency',yrange=None):
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

    assert true_momentum.shape[0] == scores.shape[0], 'Error: true_momentum must have same length as softmaxes'
    assert reconstructed_momentum.shape[0] == scores.shape[0], 'Error: reconstructed_momentum must have same length as softmaxes'

    label_size = 14

    # remove gamma events
    scores, labels, true_momentum, reconstructed_momentum = separate_particles([scores, labels, true_momentum, reconstructed_momentum],labels,index_dict,desired_labels=['$e$','$\mu$'])
    scores, labels, true_momentum, reconstructed_momentum = np.concatenate(scores), np.concatenate(labels), np.concatenate(true_momentum), np.concatenate(reconstructed_momentum)

    # bin by reconstructed momentum
    bins = [0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))]   
    bins = bins[0:-1]
    recons_mom_bin_assignments = np.digitize(reconstructed_momentum, bins)
    recons_mom_bin_idxs_list = [[]]*len(bins)

    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments == bin_num)[0]

    # compute threshold giving fixed fpr per reconstructed energy bin
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list): 
        # TODO: include bin only if shape > 0
        if bin_idxs.shape[0] > 0:
            fps, tps, thresholds = binary_clf_curve(labels[bin_idxs], scores[bin_idxs], pos_label=index_dict['$e$'])

            fns = tps[-1] - tps
            tns = fps[-1] - fps
            fprs = fps/(fps + tns)
            operating_point_idx = (np.abs(fprs - fpr_fixed_point)).argmin()
            thresholds_per_event[bin_idxs] = thresholds[operating_point_idx]

    #bin by true momentum
    ns, bins = np.histogram(true_momentum, bins=true_mom_bins, range=(200., np.max(true_momentum)) if metric=='mu fpr' else (0,1000))
    bins = bins[0:-1]
    true_mom_bin_assignments = np.digitize(true_momentum, bins)
    true_mom_bin_idxs_list = [[]]*len(bins)
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        true_mom_bin_idxs_list[bin_idx] = np.where(true_mom_bin_assignments == bin_num)[0]

    #find metrics for each true momentum bin
    bin_metrics=[]
    for bin_idxs in true_mom_bin_idxs_list:
        pred_pos_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] > 0)[0]
        pred_neg_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] < 0)[0]

        fp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['$\mu$'] )[0].shape[0]
        tp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['$e$'] )[0].shape[0]
        fn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['$e$'] )[0].shape[0]
        tn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['$\mu$'] )[0].shape[0]
        
        # TODO: division by zero
        if metric == 'efficiency':
            bin_metrics.append(tp/(tp+fn))
        else:
            bin_metrics.append(fp/(fp + tn))

    #plot metrics
    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(true_momentum) - bins[-1])/2 + bins[-1] if metric=='mu fpr' else (1000 - bins[-1])/2 + bins[-1])
    if metric == 'efficiency':
        metric_name = 'e- Signal Efficiency'
    else :
        metric_name ='\u03BC- Mis-ID Rate'
    title = '{} \n vs True Momentum At Reconstructed Momentum Bin \u03BC- Mis-ID Rate of {}%{}'.format(metric_name, fpr_fixed_point*100, title_note)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))
    
    ax.errorbar(bin_centers[:50],bin_metrics[:50],yerr=np.zeros_like(bin_metrics[:50]),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
    ax.grid(b=True, which='major', color='gray', linestyle='--')
    #nax = ax.twinx()
    ax.set_ylabel(metric_name)
    ax.set_xlabel("True Momentum (MeV/c)", fontsize=label_size)
    ax.set_title(title)

    if yrange is not None: 
        ax.set_ylim(yrange) 

    return true_momentum, thresholds_per_event


def multi_plot_multi_var_binned_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax_list, subset_labels_list,
                             label_dict, ignore_dict, threshold,muon_comparison=False, use_rejection=False, linecolors=None, line_titles=None):
    fig1, axes1 = plt.subplots(3, 3, figsize=(25, 25), facecolor='w')
    fig2, axes2 = plt.subplots(3, 3, figsize=(25*1.5, 25), facecolor='w')

    for idx, (subset_softmax, subset_labels) in enumerate(zip(subset_softmax_list, subset_labels_list)):
        plot_binned_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax, subset_labels,
                             label_dict, ignore_dict, threshold, axes1, axes2, muon_comparison, use_rejection, linecolors[idx], line_titles[idx])

def plot_multi_var_binned_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax, subset_labels,
                             label_dict, ignore_dict, threshold, axes1=None, axes2=None, muon_comparison=False, use_rejection=False, linecolor='b', line_title=None):
    if muon_comparison:
        true_label_name, false_label_name = "e/gamma", "mu"
    else:
        true_label_name, false_label_name = "$e$", "$\gamma$"
    
    if axes1 is None:
        fig1, axes1 = plt.subplots(3, 3, figsize=(25, 25), facecolor='w')
    flat_axes1 = axes1.reshape(-1)

    if axes2 is None:
        fig2, axes2 = plt.subplots(3, 3, figsize=(25*1.5, 25), facecolor='w')
        fig2.suptitle('Binned ROC curves for {}'.format(line_title), fontsize=64)

    flat_axes2 = axes2.reshape(-1)
    
    c = plt.cm.viridis(np.linspace(0,1,10))
    
    for i in range(len(energy_bins) - 1):
        try:
            fpr_list, tpr_list, thr_list, titles = [],[],[],[]
        
            # Get indices of events in this bin
            energy_bin_event_idxs = get_idxs_satisfying_bounds(subset_energies, energy_bins[i], energy_bins[i + 1])

            # Divide indices into distance to wall bins
            energy_bin_d_to_wall = subset_d_to_wall[energy_bin_event_idxs]
            
            # bin wall data
            d_to_wall_perfs = []
            for j in range(len(d_to_wall_bins) - 1):
                d_to_wall_bin_event_idxs = get_idxs_satisfying_bounds(energy_bin_d_to_wall, d_to_wall_bins[j], d_to_wall_bins[j + 1])
                idxs_for_softmax = energy_bin_event_idxs[d_to_wall_bin_event_idxs]

                # Compute roc metrics for each distance to wall bin
                d_to_wall_bin_softmax = subset_softmax[idxs_for_softmax]
                d_to_wall_bin_labels  = subset_labels[idxs_for_softmax]

                if muon_comparison:
                    collapsed_class_scores_list, collapsed_class_labels_list = multi_collapse_test_output([d_to_wall_bin_softmax], [d_to_wall_bin_labels], ignore_dict, ignore_type='$\gamma$')

                    collapsed_class_labels_list = [collapsed_class_labels - 1 for collapsed_class_labels in collapsed_class_labels_list]
                    collapsed_class_scores_list = [collapsed_class_scores[:,1:] for collapsed_class_scores in collapsed_class_scores_list]
                    
                    d_to_wall_bin_softmax = collapsed_class_scores_list[0]
                    d_to_wall_bin_labels  = collapsed_class_labels_list[0]
                
                fprs, tprs, thrs = multi_compute_roc([d_to_wall_bin_softmax], [d_to_wall_bin_labels], 
                                            true_label=label_dict[true_label_name], 
                                            false_label=label_dict[false_label_name])
                
                fprs, tprs, thrs = fprs[0], tprs[0], thrs[0]
                
                rejection = 1.0/(fprs + 1e-10)
                
                fpr_list.append(fprs) 
                tpr_list.append(tprs)
                thr_list.append(thrs)
                titles.append('Range {:.0f}-{:.0f} cm, N={}'.format(d_to_wall_bins[j], d_to_wall_bins[j + 1], len(d_to_wall_bin_labels)))
                
                # Retrieve rejection at efficiency value for each wall bin
                if use_rejection:
                    # TODO: not fully implemented
                    threshold_index = np.argmin(np.abs(tprs - threshold))
                    perf_at_threshold = rejection[threshold_index]
                else:
                    threshold_index = np.argmin(np.abs(rejection - threshold))
                    perf_at_threshold = tprs[threshold_index]

                d_to_wall_perfs.append(perf_at_threshold)
            
            ax1 = flat_axes1[i]
            
            ax1.plot(np.array((d_to_wall_bins[:-1] + d_to_wall_bins[1:]) / 2)[:-1], np.array(d_to_wall_perfs)[:-1], marker='o', linestyle='dashed', color=linecolor, label=line_title)
            ax1.set_title('Energy Range ${:.2f}-{:.2f} MeV$'.format(energy_bins[i], energy_bins[i + 1]), fontsize='24')
            ax1.set_xlabel('Distance to Wall (cm)', fontsize='24')
            ax1.set_ylabel('Efficiency at Rejection = {:.0f}'.format(threshold), fontsize='24')
            if line_title is not None:
                ax1.legend()
            

            ax2 = flat_axes2[i]
            linecolors = [c[i] for i in range(len(fpr_list))]
            linestyles = [':' for _ in range(len(fpr_list))]
            
            if muon_comparison:
                xlims, ylims = [[0.9,1.0]], [[1e0,1e5]]
                legend_loc = 'lower left'
            else:
                xlims, ylims = [[0.6,1.0]], [[1e0,1e1]]
                legend_loc = 'upper right'

            roc_fig = multi_plot_roc(fpr_list, tpr_list, thr_list,"e/gamma", "mu", 
                    fig_list=[1], xlims=xlims, ylims=ylims,
                    linestyles=linestyles,linecolors=linecolors, plot_labels=titles, 
                    title=' in Range ${:.0f}-{:.0f} MeV$'.format(energy_bins[i], energy_bins[i + 1]),
                    axes=ax2,
                    font_scale=0.8,
                    legend_loc=legend_loc,
                    show=True)
        
        except IndexError:
            pass
        
    #plt.show()


def deprecated_plot_multi_var_binned_performance(subset_energies, energy_bins, subset_d_to_wall, d_to_wall_bins, subset_softmax, subset_labels,
                             label_dict, ignore_dict, threshold, axes1=None, axes2=None, muon_comparison=False, use_rejection=False, linecolor='b', line_title=None):
    if muon_comparison:
        true_label_name, false_label_name = "e/gamma", "mu"
    else:
        true_label_name, false_label_name = "$e$", "$\gamma$"
    
    if axes1 is None:
        fig1, axes1 = plt.subplots(3, 3, figsize=(25, 25), facecolor='w')
    flat_axes1 = axes1.reshape(-1)

    if axes2 is None:
        fig2, axes2 = plt.subplots(3, 3, figsize=(25*1.5, 25), facecolor='w')
        fig2.suptitle('Binned ROC curves for {}'.format(line_title), fontsize=64)

    flat_axes2 = axes2.reshape(-1)
    
    c = plt.cm.viridis(np.linspace(0,1,10))
    
    for i in range(len(energy_bins) - 1):
        try:
            fpr_list, tpr_list, thr_list, titles = [],[],[],[]
        
            # Get indices of events in this bin
            energy_bin_event_idxs = get_idxs_satisfying_bounds(subset_energies, energy_bins[i], energy_bins[i + 1])

            # Divide indices into distance to wall bins
            energy_bin_d_to_wall = subset_d_to_wall[energy_bin_event_idxs]
            
            # bin wall data
            d_to_wall_perfs = []
            for j in range(len(d_to_wall_bins) - 1):
                d_to_wall_bin_event_idxs = get_idxs_satisfying_bounds(energy_bin_d_to_wall, d_to_wall_bins[j], d_to_wall_bins[j + 1])
                idxs_for_softmax = energy_bin_event_idxs[d_to_wall_bin_event_idxs]

                # Compute roc metrics for each distance to wall bin
                d_to_wall_bin_softmax = subset_softmax[idxs_for_softmax]
                d_to_wall_bin_labels  = subset_labels[idxs_for_softmax]

                if muon_comparison:
                    collapsed_class_scores_list, collapsed_class_labels_list = multi_collapse_test_output([d_to_wall_bin_softmax], [d_to_wall_bin_labels], ignore_dict, ignore_type='$\gamma$')

                    collapsed_class_labels_list = [collapsed_class_labels - 1 for collapsed_class_labels in collapsed_class_labels_list]
                    collapsed_class_scores_list = [collapsed_class_scores[:,1:] for collapsed_class_scores in collapsed_class_scores_list]
                    
                    d_to_wall_bin_softmax = collapsed_class_scores_list[0]
                    d_to_wall_bin_labels  = collapsed_class_labels_list[0]
                
                fprs, tprs, thrs = multi_compute_roc([d_to_wall_bin_softmax], [d_to_wall_bin_labels], 
                                            true_label=label_dict[true_label_name], 
                                            false_label=label_dict[false_label_name])
                
                fprs, tprs, thrs = fprs[0], tprs[0], thrs[0]
                
                rejection = 1.0/(fprs + 1e-10)
                
                fpr_list.append(fprs) 
                tpr_list.append(tprs)
                thr_list.append(thrs)
                titles.append('Range {:.0f}-{:.0f} cm, N={}'.format(d_to_wall_bins[j], d_to_wall_bins[j + 1], len(d_to_wall_bin_labels)))
                
                # Retrieve rejection at efficiency value for each wall bin
                if use_rejection:
                    # TODO: not fully implemented
                    threshold_index = np.argmin(np.abs(tprs - threshold))
                    perf_at_threshold = rejection[threshold_index]
                else:
                    threshold_index = np.argmin(np.abs(rejection - threshold))
                    perf_at_threshold = tprs[threshold_index]

                d_to_wall_perfs.append(perf_at_threshold)
            
            ax1 = flat_axes1[i]
            
            ax1.plot(np.array((d_to_wall_bins[:-1] + d_to_wall_bins[1:]) / 2)[:-1], np.array(d_to_wall_perfs)[:-1], marker='o', linestyle='dashed', color=linecolor, label=line_title)
            ax1.set_title('Energy Range ${:.2f}-{:.2f} MeV$'.format(energy_bins[i], energy_bins[i + 1]), fontsize='24')
            ax1.set_xlabel('Distance to Wall (cm)', fontsize='24')
            ax1.set_ylabel('Efficiency at Rejection = {:.0f}'.format(threshold), fontsize='24')
            if line_title is not None:
                ax1.legend()
            

            ax2 = flat_axes2[i]
            linecolors = [c[i] for i in range(len(fpr_list))]
            linestyles = [':' for _ in range(len(fpr_list))]
            
            if muon_comparison:
                xlims, ylims = [[0.9,1.0]], [[1e0,1e5]]
                legend_loc = 'lower left'
            else:
                xlims, ylims = [[0.6,1.0]], [[1e0,1e1]]
                legend_loc = 'upper right'

            roc_fig = multi_plot_roc(fpr_list, tpr_list, thr_list,"e/gamma", "mu", 
                    fig_list=[1], xlims=xlims, ylims=ylims,
                    linestyles=linestyles,linecolors=linecolors, plot_labels=titles, 
                    title=' in Range ${:.0f}-{:.0f} MeV$'.format(energy_bins[i], energy_bins[i + 1]),
                    axes=ax2,
                    font_scale=0.8,
                    legend_loc=legend_loc,
                    show=True)
        except IndexError:
            pass
        
    #plt.show()

def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    '''
    #########################################################################
    This functionality added as det_curve in sklearn v0.24
    #########################################################################

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
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(8,8))
    bin_range = [[np.min([np.min(dist_1_x),np.min(dist_2_x)]),np.max([np.max(dist_1_x),np.max(dist_2_x)])],
             [np.min([np.min(dist_1_y),np.min(dist_2_y)]),np.max([np.max(dist_1_y),np.max(dist_2_y)])]]
    ns_1, xedges, yedges = np.histogram2d(dist_1_x,dist_1_y,bins=bins,density=True,range=bin_range)
    ns_2,_,_ = np.histogram2d(dist_2_x,dist_2_y,bins=bins,density=True,range=bin_range)
    ratio = ns_1/ns_2
    ratio = np.where((ns_2==0) & (ns_1==0),1,ratio)
    ratio = np.where((ns_2==0) & (ns_1!=0),10,ratio)
    pc = ax.pcolormesh(xedges, yedges, np.swapaxes(ratio,0,1),vmin=ratio_range[0],vmax=ratio_range[1],cmap="RdBu_r")
    fig.colorbar(pc, ax=ax)
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig

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
    fig, axes = plt.subplots(len(particle_names),noutputs+len(extra_panes),figsize=(12*len(particle_names),12*noutputs+len(extra_panes)))

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