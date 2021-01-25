import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from functools import reduce

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,2), #5
           (0,4), #6
           (1,6), #7
           (2,8), #8
           (4,8), #9
           (6,8), #10
           (7,6), #11
           # Inner ring
           (6,4), #12
           (5,2), #13
           (3,2), #14
           (2,4), #15
           (3,6), #16
           (5,6), #17
           (4,4)] #18

PADDING = 1

def get_plot_array(event_data):
    
    # Assertions on the shape of the data and the number of input channels
    assert(len(event_data.shape) == 3 and event_data.shape[2] == 19)
    
    # Extract the number of rows and columns from the event data
    rows = event_data.shape[0]
    cols = event_data.shape[1]
    
    # Make empty output pixel grid
    output = np.empty(((10+PADDING)*rows, (10+PADDING)*cols))
    output[:] = np.nan
    i, j = 0, 0
    
    for row in range(rows):
        j = 0
        for col in range(cols):
            pmts = event_data[row, col]
            tile(output, (i, j), pmts)
            j += 10 + PADDING
        i += 10 + PADDING
        
    return output

def tile(canvas, ul, pmts):
    
    # First, create 10x10 grid representing single mpmt
    mpmt = np.empty((10, 10))
    mpmt[:] = np.nan
    for i, val in enumerate(pmts):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val

    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]

def disp_learn_hist(location, losslim=None, axis=None, show=True):
    
    """
    Purpose : Plot the loss and accuracy history for a training session
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """
    val_log=location + '/log_val.csv'
    val_log_df  = pd.read_csv(val_log)

    train_log_df = get_aggregated_train_data(location)

    if axis is None:
        fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    else:
        ax1 = axis
    
    line11 = ax1.plot(train_log_df.epoch, train_log_df.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(val_log_df.epoch, val_log_df.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log_df.epoch, train_log_df.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(val_log_df.epoch, val_log_df.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.05)

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return
    
    if axis is None:
        return fig

def get_aggregated_train_data(location):
    # get all training data files
    base_log_path = location + '/log_train_[0-9]*.csv'
    log_paths = glob.glob(base_log_path)

    print("Found training logs: ", log_paths)
    
    log_dfs = []
    for log_path in log_paths:
        log_dfs.append(pd.read_csv(log_path))
        log_dfs.append(pd.read_csv(log_path))
    
    # combine all files into one dataframe
    train_log_df = pd.DataFrame(0, index=np.arange(len(log_dfs[0])), columns=log_dfs[0].columns)
    for idx, df_vals in enumerate(zip(*[log_df.values for log_df in log_dfs])):
        iteration = df_vals[0][0]
        epoch = df_vals[0][1]
        loss = sum([df_val[2] for df_val in df_vals]) / len(df_vals)
        accuracy = sum([df_val[3] for df_val in df_vals]) / len(df_vals)

        output_df_vals = (iteration, epoch, loss, accuracy)
        train_log_df.iloc[idx] = output_df_vals

    return train_log_df

def disp_learn_hist_smoothed(location, losslim=None, window_train=400, window_val=40, show=True):
    """
    Purpose : Plot the loss and accuracy history for a training session with averaging to clean up noise
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """
    val_log=location + '/log_val.csv'
    val_log_df   = pd.read_csv(val_log)

    train_log_df = get_aggregated_train_data(location)

    epoch_train    = moving_average(np.array(train_log_df.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_df.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_df.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_df.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_df.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_df.loss),window_val)

    epoch_val_uns    = np.array(val_log_df.epoch)
    accuracy_val_uns = np.array(val_log_df.accuracy)
    loss_val_uns     = np.array(val_log_df.loss)

    saved_best      = np.array(val_log_df.saved_best)
    stored_indices  = np.where(saved_best>1.0e-3)
    epoch_val_st    = epoch_val_uns[stored_indices]
    accuracy_val_st = accuracy_val_uns[stored_indices]
    loss_val_st     = loss_val_uns[stored_indices]

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(epoch_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch_val, loss_val, label='Average validation loss', color='blue')
    line13 = ax1.scatter(epoch_val_st, loss_val_st, label='BEST validation loss',
                         facecolors='none', edgecolors='blue',marker='o')
    
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(epoch_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch_val, accuracy_val, label='Average validation accuracy', color='red')
    line23 = ax2.scatter(epoch_val_st, accuracy_val_st, label='BEST accuracy',
                         facecolors='none', edgecolors='red',marker='o')
    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)
    
    # added these four lines
    lines  = line11+ line12+ [line13]+ line21+ line22+ [line23]
    #lines_sctr=[line13,line23]
    #lines=lines_plt+lines_sctr

    labels = [l.get_label() for l in lines]
    
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return

    return fig

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, class_names):
    
    """
    plot_confusion_matrix(labels, predictions, class_names)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
    """
    
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')

    num_labels = len(class_names)
    max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])

    assert max_value < num_labels

    mat,_,_,im = ax.hist2d(predictions, labels,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

    # Normalize the confusion matrix
    mat = mat.astype("float") / mat.sum(axis=0)#[:, np.newaxis]

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20) 
        
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=20)
    ax.set_yticklabels(class_names,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.title("Confusion matrix", fontsize=20) 
   
    plt.show()

def plot_classifier_response(softmaxes, labels, particle_names, label_dict,linestyles=None,bins=None,legend_locs=None,
                  extra_panes=[], xlim=None,label_size=14, legend_label_dict=None, show=True):
    '''
    Plots classifier softmax outputs for each particle type.
    Args:
        softmaxes                    ... 2d array with first dimension n_samples
        labels                       ... 1d array of particle labels to use in every output plot, or list of 4 lists of particle names to use in each respectively
        particle_names               ... list of string names of particle types to plot. All must be keys in 'label_dict' 
        label_dict                   ... dictionary of particle labels, with string particle name keys and values corresponsing to 
                                         values taken by 'labels'
        bins                         ... optional, number of bins for histogram
        fig, axes                    ... optional, figure and axes on which to do plotting (use to build into bigger grid)
        legend_locs                  ... list of 4 strings for positioning the legends
        extra_panes                  ... list of lists of particle names, each of which contains the names of particles to use in a joint response plot
        xlim                         ... limit the x-axis
        label_size                   ... font size
        legend_label_dict            ... dictionary of display symbols for each string label, to use for displaying pretty characters
    author: Calum Macdonald
    June 2020
    '''
    if legend_label_dict is None:
        legend_label_dict={}
        for name in particle_names:
            legend_label_dict[name] = name
    
    legend_size=label_size

    num_panes = softmaxes.shape[1]+len(extra_panes)

    fig, axes = plt.subplots(1,num_panes,figsize=(5*num_panes,5))
    inverse_label_dict = {value:key for key, value in label_dict.items()}

    softmaxes_list = separate_particles([softmaxes], labels, label_dict, [name for name in label_dict.keys()])[0]

    if isinstance(particle_names[0],str):
        particle_names = [particle_names for _ in range(num_panes)]

    # generate single particle plots
    for independent_particle_label, ax in enumerate(axes[:softmaxes.shape[1]]):
        print(label_dict)
        dependent_particle_labels = [label_dict[particle_name] for particle_name in particle_names[independent_particle_label]]
        for dependent_particle_label in dependent_particle_labels:
            ax.hist(softmaxes_list[dependent_particle_label][:,independent_particle_label],
                    label=f"{legend_label_dict[inverse_label_dict[dependent_particle_label]]} Events",
                    alpha=0.7,histtype=u'step',bins=bins,density=True,
                    linestyle=linestyles[dependent_particle_label],linewidth=2)            
        ax.legend(loc=legend_locs[independent_particle_label] if legend_locs is not None else 'best', fontsize=legend_size)
        ax.set_xlabel('P({})'.format(legend_label_dict[inverse_label_dict[independent_particle_label]]), fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')

    ax = axes[-1]

    # generate joint plots
    for n, extra_pane_particle_names in enumerate(extra_panes):
        pane_idx = softmaxes.shape[1]+n
        ax=axes[pane_idx]
        dependent_particle_labels = [label_dict[particle_name] for particle_name in particle_names[pane_idx]]
        for dependent_particle_label in dependent_particle_labels:
                ax.hist(reduce(lambda x,y : x+y, [softmaxes_list[dependent_particle_label][:,label_dict[pname]] for pname in extra_pane_particle_names]),
                        label=legend_label_dict[particle_names[-1][dependent_particle_label]],
                        alpha=0.7,histtype=u'step',bins=bins,density=True,
                        linestyle=linestyles[dependent_particle_label],linewidth=2)         
        ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=legend_size)
        xlabel = ''
        for list_index, independent_particle_name in enumerate(extra_pane_particle_names):
            xlabel += 'P({})'.format(legend_label_dict[independent_particle_name])
            if list_index < len(extra_pane_particle_names) - 1:
                xlabel += ' + '
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')
    
    plt.tight_layout()

    if show:
        plt.show()
        return

    return fig

def separate_particles(input_array_list,labels,index_dict,desired_labels=['gamma','e','mu']):
    '''
    Separates all arrays in a list by indices where 'labels' takes a certain value, corresponding to a particle type.
    Assumes that the arrays have the same event order as labels. Returns list of tuples, each tuple contains section of each
    array corresponsing to a desired label.
    Args:
        input_array_list            ... list of arrays to be separated, must have same length and same length as 'labels'
        labels                      ... list of labels, taking any of the three values in index_dict.values()
        index_dict                  ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels', 
                                        unless desired_labels is passed
        desired_labels              ... optional list specifying which labels are desired and in what order. Default is ['gamma','e','mu']
    author: Calum Macdonald
    June 2020
    '''
    idxs_list = [np.where(labels==index_dict[label])[0] for label in desired_labels]

    separated_arrays = []
    for array in input_array_list:
        separated_arrays.append(tuple([array[idxs] for idxs in idxs_list]))

    return separated_arrays

def compute_roc(softmax_out_val, labels_val, true_label, false_label):
    # Compute ROC metrics
    labels_val_for_comp = labels_val[np.where( (labels_val==false_label) | (labels_val==true_label)  )]
    softmax_out_for_comp = softmax_out_val[np.where(  (labels_val==false_label) | (labels_val==true_label)  )][:,true_label]

    fpr, tpr, thr = roc_curve(labels_val_for_comp, softmax_out_for_comp, pos_label=true_label)
    
    return fpr, tpr, thr

def plot_roc(fpr, tpr, thr, true_label_name, false_label_name, fig_list=None, xlims=None, ylims=None, axes=None, linestyle=None, linecolor=None, plot_label=None, show=False):
    """
    Purpose : Plot ROC curves for a classifier that has been evaluated on a validation set with respect to given labels
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """
    # Compute additional parameters
    rejection=1.0/(fpr+1e-10)
    roc_AUC = auc(fpr,tpr)

    if fig_list is None:
        fig_list = list(range(3))
    
    figs = []
    # Plot results
    if axes is None:
        if 0 in fig_list:
            fig0, ax0 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig0)
        if 1 in fig_list: 
            fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig1)
        if 2 in fig_list: 
            fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
            figs.append(fig2)
    else:
        print(axes)
        axes_iter = iter(axes)
        if 0 in fig_list:
            ax0 = next(axes_iter)
        if 1 in fig_list: 
            ax1 = next(axes_iter)
        if 2 in fig_list: 
            ax2 = next(axes_iter)

    if xlims is not None:
        xlim_iter = iter(xlims)
    if ylims is not None:
        ylim_iter = iter(ylims)

    if 0 in fig_list: 
        ax0.tick_params(axis="both", labelsize=20)
        ax0.plot(fpr, tpr,
                    label=plot_label if plot_label + ', AUC={:.3f}'.format(roc_AUC)  is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor if linecolor is not None else None)
        ax0.set_xlabel('FPR', fontsize=20)
        ax0.set_ylabel('TPR', fontsize=20)
        ax0.legend(loc="lower right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax0.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax0.set_ylim(ylim[0],ylim[1])
    
    if 1 in fig_list: 
        ax1.tick_params(axis="both", labelsize=20)
        ax1.set_yscale('log')
        ax1.grid(b=True, which='major', color='gray', linestyle='-')
        ax1.grid(b=True, which='minor', color='gray', linestyle='--')
        ax1.plot(tpr, rejection, 
                    label=plot_label + ', AUC={:.3f}'.format(roc_AUC)  if plot_label is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor if linecolor is not None else None)

        xlabel = f'{true_label_name} Signal Efficiency'
        ylabel = f'{false_label_name} Background Rejection'
        title = '{} vs {} Rejection'.format(true_label_name, false_label_name)

        ax1.set_xlabel(xlabel, fontsize=20)
        ax1.set_ylabel(ylabel, fontsize=20)
        ax1.set_title(title, fontsize=24)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #loc="upper right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax1.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax1.set_ylim(ylim[0],ylim[1])
    
    if 2 in fig_list: 
        ax2.tick_params(axis="both", labelsize=20)
        #plt.yscale('log')
        #plt.ylim(1.0,1)
        ax2.grid(b=True, which='major', color='gray', linestyle='-')
        ax2.grid(b=True, which='minor', color='gray', linestyle='--')
        ax2.plot(tpr, tpr/np.sqrt(fpr), 
                    label= plot_label + ', AUC={:.3f}'.format(roc_AUC) if plot_label is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC),
                    linestyle=linestyle  if linestyle is not None else None,
                    color=linecolor if linecolor is not None else None)
        ax2.set_xlabel('efficiency', fontsize=20)
        ax2.set_ylabel('~significance', fontsize=20)
        ax2.legend(loc="upper right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax2.set_xlim(xlim[0],xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax2.set_ylim(ylim[0],ylim[1])

    if show:
        plt.show()
        return
    
    if axes is None:
        return tuple(figs)

def plot_rocs(softmax_out_val, labels_val, labels_dict, plot_list=None, vs_list = None, show=True):
    
    """
    Purpose : Plot ROC curves for a classifier that has been evaluated on a validation set for a series of combinations of labels
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """
    # Compute ROC metrics
    # if no list of labels to plot, assume using all members of dict
    all_labels = list(labels_dict.keys())

    if plot_list is None:
        plot_list = all_labels
    
    if vs_list is None:
        vs_list = all_labels
    
    figs = []
    # plot ROC curves for each specified label
    for true_label_name in plot_list:
        true_label = labels_dict[true_label_name]
        for false_label_name in vs_list:
            false_label = labels_dict[false_label_name]
            if not (false_label_name == true_label_name):
                # initialize figure
                num_panes = 3
                fig, axes = plt.subplots(1, num_panes, figsize=(8*num_panes,12))
                fig.suptitle("ROC for {} vs {}".format(true_label_name, false_label_name), fontweight='bold',fontsize=32)

                plot_roc(softmax_out_val, labels_val, true_label_name, true_label, false_label_name, false_label, axes=axes)
                
                figs.append(fig)
    if show:
        plt.show()
        return
    
    return figs
