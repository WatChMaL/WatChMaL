import matplotlib.pyplot as plt

from analysis.plot_utils import get_aggregated_train_data

def disp_learn_hist(location, losslim=None, axis=None, show=True):
    """
    Purpose : Plot the loss and accuracy history for a training session
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """

    train_log_df = get_aggregated_train_data(location)

    if axis is None:
        fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    else:
        ax1 = axis
    
    line11 = ax1.plot(train_log_df.epoch, train_log_df.g_loss, linewidth=2, label='Generator Loss', color='g', alpha=0.3)
    line12 = ax1.plot(train_log_df.epoch, train_log_df.d_loss_fake, linewidth=2, label='Discriminator Fake Data Loss', color='o', alpha=0.3)
    line13 = ax1.plot(train_log_df.epoch, train_log_df.d_loss_real, linewidth=2, label='Discriminator Real Data Loss', color='b', alpha=0.3)

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    # added these four lines
    lines  = line11 + line12 + line13
    labels = [l.get_label() for l in lines]
    leg    = ax1.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return
    
    if axis is None:
        return fig