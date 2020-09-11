import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def disp_learn_hist(location,losslim=None,show=True):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'

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

# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, energies, class_names, min_energy=0, max_energy=1500, 
                          show_plot=False, save_path=None):
    
    """
    plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, save_path=None)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          energies            ... 1D array of event energies, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
          min_energy          ... Minimum energy for the events to consider
          max_energy          ... Maximum energy for the events to consider
          show_plot[optional] ... Boolean to determine whether to display the plot
          save_path[optional] ... Path to save the plot as an image
    """
    
    # Create a mapping to extract the energies in
    energy_slice_map = [False for i in range(len(energies))]
    for i in range(len(energies)):
        if(energies[i] >= min_energy and energies[i] < max_energy):
                energy_slice_map[i] = True
                
    # Filter the CNN outputs based on the energy intervals
    labels = labels[energy_slice_map]
    predictions = predictions[energy_slice_map]
    
    if(show_plot or save_path is not None):
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
        plt.title("Confusion matrix, " + r"${0} \leq E < {1}$".format(min_energy, max_energy), fontsize=20) 
   
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the plot frame
        plt.close() # Close the opened window if any