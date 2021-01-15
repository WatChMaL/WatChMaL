import glob
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def disp_gan_learn_hist(location, losslim=None, xlim=None, axis=None, show=True):
    """
    Purpose : Plot the loss and accuracy history for a training session
    
    Args: location     ... output directory containing log files
          losslim      ... sets bound on y axis of loss
          show         ... if true then display figure, otherwise return figure
    """

    train_log_df = gan_get_aggregated_train_data(location)

    if axis is None:
        fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    else:
        ax1 = axis
    
    line11 = ax1.plot(train_log_df.epoch, train_log_df.g_loss, linewidth=2, label='Generator Loss', color='g', alpha=0.3)
    line12 = ax1.plot(train_log_df.epoch, train_log_df.d_loss_fake, linewidth=2, label='Discriminator Fake Data Loss', color='orange', alpha=0.3)
    line13 = ax1.plot(train_log_df.epoch, train_log_df.d_loss_real, linewidth=2, label='Discriminator Real Data Loss', color='b', alpha=0.3)

    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    if xlim is not None:
        ax1.set_xlim(xlim)

    # added these four lines
    lines  = line11 + line12 + line13
    labels = [l.get_label() for l in lines]
    leg    = ax1.legend(lines, labels, fontsize=16, numpoints=1, loc='upper right')
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return
    
    if axis is None:
        return fig

def gan_get_aggregated_train_data(location):
    # TODO: fix overlap with method in plot_utils
    # get all training data files
    base_log_path = location + '/log_train_[0-9]*.csv'
    log_paths = glob.glob(base_log_path)

    print("Found training logs: ", log_paths)
    
    log_dfs = []
    for log_path in log_paths:
        log_dfs.append(pd.read_csv(log_path))
        log_dfs.append(pd.read_csv(log_path))
    
    # combine all files into one dataframe
    print(log_dfs[0].columns)
    train_log_df = pd.DataFrame(0, index=np.arange(len(log_dfs[0])), columns=log_dfs[0].columns)
    for idx, df_vals in enumerate(zip(*[log_df.values for log_df in log_dfs])):
        iteration = df_vals[0][0]
        epoch = df_vals[0][1]
        g_loss = sum([df_val[2] for df_val in df_vals]) / len(df_vals)
        d_fake_loss = sum([df_val[3] for df_val in df_vals]) / len(df_vals)
        d_real_loss = sum([df_val[4] for df_val in df_vals]) / len(df_vals)

        output_df_vals = (iteration, epoch, g_loss, d_fake_loss, d_real_loss)
        train_log_df.iloc[idx] = output_df_vals

    return train_log_df

def load_image_batches(path):
    """
    Load image batches produced by generator
    """
    return [np.load(fname, allow_pickle=True)['gen_imgs'] for fname in glob.glob(os.path.join(path,'imgs/*'))]


# ========================================================================
# Collapsed GAN plotting functions

def display_np_collapsed_batch(image_batch, fig):
    """
    Display collapsed image batch using numpy format
    """
    batch_grid = np.concatenate([np.pad(np.squeeze(img),2,mode='constant',constant_values=(np.nan,)) for img in image_batch[0:8]], axis=1)
    
    for row_index in range(1,8,1):
        image_row = np.concatenate([np.pad(np.squeeze(img),2,mode='constant',constant_values=(np.nan,)) for img in image_batch[row_index*8:(row_index+1)*8]],axis=1)
        batch_grid = np.concatenate((batch_grid, image_row),axis=0)
    #plt.figure(figsize=(25,25))
    im = plt.imshow(batch_grid, animated=True,cmap=plt.cm.viridis)
    return [im]

def animate_np_collapsed_batches(image_batches):
    """
    Animate collapsed image batches using numpy format
    """
    fig = plt.figure(figsize=(25,25))
    plt.axis("off")
    ims = []
    for im in image_batches:
        ims.append(display_np_collapsed_batch(im, fig))
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    return HTML(ani.to_jshtml())

def display_collapsed_batch(image_batch, axes):
    """
    Display collapsed image batches in higher resolution plt format
    """
    ims = []
    for idx, ax in enumerate(axes.reshape(-1)):
        im = ax.imshow(image_batch[idx, 0, :, :], interpolation='nearest', animated=True)
        ims.append(im)
    return ims

def animate_collapsed_batches(image_batches):
    """
    Animate collapsed image batches in higher resolution plt format
    """
    fig, axes = plt.subplots(8, 8, figsize=(25, 25))
    list_of_frames = []
    for image_batch in image_batches:
        list_of_frames.append(display_collapsed_batch(image_batch, axes))
    ani = animation.ArtistAnimation(fig, list_of_frames, interval=1000, repeat_delay=1000, blit=True)
    return HTML(ani.to_jshtml())

# ========================================================================
# Full 19 Channel GAN plotting functions
