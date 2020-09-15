import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def disp_learn_hist(location,losslim=None,show=True):
    
    """
    Purpose : Plot the loss and accuracy history for a training session
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """
    train_log=location + '/log_train.csv'
    val_log=location + '/log_val.csv'

    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(train_log_csv.epoch, train_log_csv.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log_csv.epoch, train_log_csv.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

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
    
    return fig

def disp_learn_hist_smoothed(location, losslim=None, window_train=400,window_val=40,show=True):
    train_log=location + '/log_train.csv'
    val_log=location + '/log_val.csv'
    
    train_log_csv = pd.read_csv(train_log)
    val_log_csv   = pd.read_csv(val_log)

    epoch_train    = moving_average(np.array(train_log_csv.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_csv.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_csv.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_csv.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_csv.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_csv.loss),window_val)

    epoch_val_uns    = np.array(val_log_csv.epoch)
    accuracy_val_uns = np.array(val_log_csv.accuracy)
    loss_val_uns     = np.array(val_log_csv.loss)

    saved_best      = np.array(val_log_csv.saved_best)
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
    mat = mat.astype("float") / mat.sum(axis=0)[:, np.newaxis]

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

def plot_classifier_response(softmaxes, labels, particle_names, index_dict,linestyles=None,bins=None,legend_locs=None,fitqun=False,
                  extra_panes=[], xlim=None,label_size=14, legend_label_dict=None, show=True):
    '''
    Plots classifier softmax outputs for each particle type.
    Args:
        softmaxes                    ... 2d array with first dimension n_samples
        labels                       ... 1d array of particle labels to use in every output plot, or list of 4 lists of particle names to use in each respectively
        particle_names               ... list of string names of particle types to plot. All must be keys in 'index_dict' 
        index_dict                   ... dictionary of particle labels, with string particle name keys and values corresponsing to 
                                         values taken by 'labels'
        bins                         ... optional, number of bins for histogram
        fig, axes                    ... optional, figure and axes on which to do plotting (use to build into bigger grid)
        legend_locs                  ... list of 4 strings for positioning the legends
        fitqun                       ... designate if the given scores are from fitqun
        xlim                        ... limit the x-axis
        label_size                  ... font size
        legend_label_dict           ... dictionary of display symbols for each string label, to use for displaying pretty characters
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
    label_dict = {value:key for key, value in index_dict.items()}

    softmaxes_list = separate_particles([softmaxes], labels, index_dict, [name for name in index_dict.keys()])[0]

    if isinstance(particle_names[0],str):
        particle_names = [particle_names for _ in range(num_panes)]

    for output_idx,ax in enumerate(axes[:softmaxes.shape[1]]):
        for i in [index_dict[particle_name] for particle_name in particle_names[output_idx]]:
            ax.hist(softmaxes_list[i][:,output_idx],
                    label=f"{legend_label_dict[label_dict[i]]} Events",
                    alpha=0.7,histtype=u'step',bins=bins,density=True,
                    linestyle=linestyles[i],linewidth=2)            
        ax.legend(loc=legend_locs[output_idx] if legend_locs is not None else 'best', fontsize=legend_size)
        ax.set_xlabel('P({})'.format(legend_label_dict[label_dict[output_idx]]), fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')
    ax = axes[-1]
    for n, extra_pane_particle_names in enumerate(extra_panes):
        pane_idx = softmaxes.shape[1]+n
        ax=axes[pane_idx]
        for i in [index_dict[particle_name] for particle_name in particle_names[pane_idx]]:
                ax.hist(reduce(lambda x,y : x+y, [softmaxes_list[i][:,index_dict[pname]] for pname in extra_pane_particle_names]),
                        label=legend_label_dict[particle_names[-1][i]],
                        alpha=0.7,histtype=u'step',bins=bins,density=True,
                        linestyle=linestyle[i],linewidth=2)         
        ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=legend_size)
        ax.set_xlabel('P({}) + P({})'.format(legend_label_dict['gamma'],legend_label_dict['e']), fontsize=label_size)
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

def plot_rocs(softmax_out_val, labels_val, labels_dict, plot_list=None, show=True):
    # if no list of labels to plot, assume plotting all
    all_labels = list(labels_dict.keys())
    if plot_list is None:
        plot_list = all_labels
    figs = []
    # plot ROC curves for each specified label
    for true_label_name in plot_list:
        true_label = labels_dict[true_label_name]
        
        for false_label_name in all_labels:
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

def plot_roc(softmax_out_val, labels_val, true_label_name, true_label, false_label_name, false_label, axes=None, show=False):
    # Compute ROC metrics
    labels_val_for_comp = labels_val[np.where( (labels_val==false_label) | (labels_val==true_label)  )]
    softmax_out_for_comp = softmax_out_val[np.where(  (labels_val==false_label) | (labels_val==true_label)  )][:,true_label]

    fpr, tpr, thr = roc_curve(labels_val_for_comp, softmax_out_for_comp, pos_label=true_label)
    
    roc_AUC = auc(fpr,tpr)

    # Plot results
    if axes is None:
        fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
        fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
        fig3, ax3 = plt.subplots(figsize=(12,8),facecolor="w")
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

    ax1.tick_params(axis="both", labelsize=20)
    ax1.plot(fpr,tpr,label=r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC))
    ax1.set_xlabel('FPR',fontweight='bold',fontsize=24,color='black')
    ax1.set_ylabel('TPR',fontweight='bold',fontsize=24,color='black')
    ax1.legend(loc="lower right",prop={'size': 16})

    rejection=1.0/(fpr+1e-10)
    
    ax2.tick_params(axis="both", labelsize=20)
    ax2.set_yscale('log')
    ax2.set_ylim(1.0,1.0e3)
    ax2.grid(b=True, which='major', color='gray', linestyle='-')
    ax2.grid(b=True, which='minor', color='gray', linestyle='--')
    ax2.plot(tpr, rejection, label=r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC))
    ax2.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
    ax2.set_ylabel('Rejection',fontweight='bold',fontsize=24,color='black')
    ax2.legend(loc="upper right",prop={'size': 16})
    
    ax3.tick_params(axis="both", labelsize=20)
    #plt.yscale('log')
    #plt.ylim(1.0,1)
    ax3.grid(b=True, which='major', color='gray', linestyle='-')
    ax3.grid(b=True, which='minor', color='gray', linestyle='--')
    ax3.plot(tpr, tpr/np.sqrt(fpr), label=r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name, roc_AUC))
    ax3.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
    ax3.set_ylabel('~significance',fontweight='bold',fontsize=24,color='black')
    ax3.legend(loc="upper right",prop={'size': 16})

    if show:
        plt.show()
        return
    
    if axes is None:
        return fig1, fig2, fig3
