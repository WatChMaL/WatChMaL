from plot_utils import disp_learn_hist, plot_confusion_matrix, disp_learn_hist_smoothed, plot_classifier_response, plot_roc, plot_rocs
import numpy as np
import matplotlib.pyplot as plt

def main():
    ############################################################
    # TEST MAIN RUNS
    #loc = "../outputs/2020-09-15/resnet/outputs/20200915" # Resnet
    #loc = "../outputs/2020-09-15/pointnet/outputs/20200915" # pointnet

    loc = "../outputs/2020-09-16/10-27-57/outputs/20200916" # New Resnet


    disp_learn_hist(loc)
    #disp_learn_hist_smoothed(loc)

    ############################################################
    
    # display learning
    
    # old
    # loc = "../outputs/2020-09-11/14-07-18/outputs/20200911"

    # new
    # loc = '/home/jtindall/WatChMaL/outputs/2020-09-14/14-20-18/outputs/20200914'

    # Big pointnet
    # loc = '/home/jtindall/WatChMaL/outputs/2020-09-14/18-26-22/outputs/20200914'

    # Big resnet (with np arrays)
    #loc = '/home/jtindall/WatChMaL/outputs/2020-09-14/20-35-56/outputs/20200914'

    # updated pointnet
    # loc = '/home/jtindall/WatChMaL/outputs/2020-09-15/09-49-54/outputs/20200915'

    # smoothed resnet
    #loc = '/home/jtindall/WatChMaL/outputs/2020-09-15/10-45-14/outputs/20200915'

    # bigger pointnet
    # loc = '/home/jtindall/WatChMaL/outputs/2020-09-15/11-26-51/outputs/20200915'
    # disp_learn_hist(loc)
    

    # display smoothed learning
    #disp_learn_hist_smoothed(loc)
    

    # display confusion matrix
    #loc = "../outputs/2020-09-11/14-07-18/outputs"
    
    #loc = '../outputs/2020-09-14/20-35-56/outputs/20200914'

    labels_val=np.load(loc + "/labels.npy")
    predictions_val=np.load(loc + "/predictions.npy")
    softmax_out_val=np.load(loc + "/softmax.npy")

    #plot_confusion_matrix(labels_val, predictions_val, ['$\gamma$','$e$','$\mu$'])

    # plot classifier responses
    label_dict = {"$\gamma$":0, "$e$":1, "$\mu$":2}
    plot_classifier_response(softmax_out_val, labels_val, particle_names=['$\gamma$','$e$','$\mu$'], label_dict=label_dict, linestyles=(':','-','--'), extra_panes =[['$\gamma$','$e$']], bins=30)

    # plot ROC curves
    #prep_roc_data(softmax_out_val,labels_val, 'rejection', index_dict, "$e$","$\gamma$")

    # plot_original_roc(softmax_out_val,labels_val, show=True)
    #plot_roc(softmax_out_val,labels_val, "$e$", label_dict["$e$"], "$\gamma$", label_dict["$\gamma$"], show=True)
    #plot_rocs(softmax_out_val,labels_val, label_dict, plot_list = ["$e$"])

    #fig1, fig2, fig3 = plot_roc(softmax_out_val,labels_val, "$e$", label_dict["$e$"], "$\gamma$", label_dict["$\gamma$"], show=False)
    #fig1, fig2, fig3 = plot_roc(softmax_out_val,labels_val, "$e$", label_dict["$e$"], "$\mu$", label_dict["$\mu$"], show=False)
    plt.show()
    


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
