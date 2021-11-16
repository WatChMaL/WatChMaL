import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

np.set_printoptions(threshold=np.inf)


# All data: angles, energies, hit_charge, hit_pmt, hit_time, labels, positions, root_files, veto, veto2
h5File = "/fast_scratch/WatChMaL/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emg_E0to1000MeV_digihits.h5"

# Image positions::
npFile = "/home/jyang/project/regression-project/outputs/2021-11-03/12-47-21/outputs/predictions.npy"

# Split for training, validation, and test
splitPathFile = "/home/ykwon/electron.npz"

# Utility function to read npz file for exploration
def readNpzFile(npz_file_path):
    data = np.load(npz_file_path)
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

def readNpyFile(npy_file):
    data = np.load(npy_file)
    print(data[:1000])

def readH5File(h5_file_path):
    f = h5py.File(h5_file_path, 'r')
    print(f.keys())
    dataset = f['energies'][()]

    dataset.sort()
    print(type(dataset))

def plot_energy_histogram():
    location = '/home/jyang/project/regression-project/outputs/first-energy-run-2021-10-04/first-energy-run-15-15-42/outputs'
    ypred_loc = location + '/predictions.npy'
    yind_loc = location + '/indices.npy'
    f = h5py.File('/fast_scratch/WatChMaL/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emg_E0to1000MeV_digihits.h5', 'r')
    print(list(f.keys()))

    # print(ypred.describe())

    indices = np.load(yind_loc).flatten()
    test_idxs = np.load('/home/ykwon/electron.npz')['test_idxs']
    e_pred = np.load(ypred_loc).flatten()
    e_actual = np.array(f['energies']).flatten()[test_idxs][indices]
    abs_diff = (e_actual-e_pred)/e_actual
    plt.hist(abs_diff, histtype='step', bins=200, label='Difference', range=(-2, 2))
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.savefig(location + '/scaled_hist.png')

def plot_position_histogram():
    location = '/home/jyang/project/regression-project/outputs/2021-11-03/12-47-21/outputs'
    ypred_loc = location + '/predictions.npy'
    yind_loc = location + '/indices.npy'
    f = h5py.File('/fast_scratch/WatChMaL/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emg_E0to1000MeV_digihits.h5', 'r')
    print(f.keys())
    indices = np.load(yind_loc).flatten()
    test_idxs = np.load('/home/ykwon/electron.npz')['test_idxs']
    e_pred = np.load(ypred_loc)
    e_actual = np.array([np.array(x[0]) for x in np.array(f['positions'])[test_idxs][indices]])
    #print(e_pred[:5])

    predicted_tranverse = np.array([np.sqrt(np.power(x[0], 2) + np.power(x[2], 2)) for x in e_pred])
    actual_tranverse = np.array([np.sqrt(np.power(x[0], 2) + np.power(x[2], 2)) for x in e_actual])

    x_pred = np.array([x[0] for x in e_pred])
    y_pred = np.array([x[1] for x in e_pred])
    z_pred = np.array([x[2] for x in e_pred])

    x_actual = np.array([x[0] for x in e_actual])
    y_actual = np.array([x[1] for x in e_actual])
    z_actual = np.array([x[2] for x in e_actual])
    
    tranverse_diff = abs(actual_tranverse - predicted_tranverse)
    x_diff = (x_actual-x_pred)
    y_diff = (y_actual-y_pred)
    z_diff = (z_actual-z_pred)

    three_dimension_difference = np.sqrt(np.square(np.subtract(x_actual, x_pred)) + np.square(np.subtract(y_actual, y_pred)) + np.square(np.subtract(z_actual, z_pred)))
    
    plt.figure()
    plt.hist(tranverse_diff, histtype='step', bins=200, label='Difference')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Tranverse Prediction Errors')
    plt.savefig(location + '/transverse_position_scaled_hist.png')

    plt.figure()
    plt.hist(x_diff, histtype='step', bins=200, label='Difference', range=(-200, 200))
    plt.xlabel('Distance to True Vertex (cm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of X-axis Prediction Errors')
    plt.savefig(location + '/x_axis_position_scaled_hist.png')

    plt.figure()
    plt.hist(y_diff, histtype='step', bins=200, label='Difference', range=(-200, 200))
    plt.xlabel('Distance to True Vertex (cm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Y-axis Prediction Errors')
    plt.savefig(location + '/y_axis_position_scaled_hist.png')

    plt.figure()
    plt.hist(z_diff, histtype='step', bins=200, label='Difference', range=(-200, 200))
    plt.xlabel('Distance to True Vertex (cm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Z-axis Prediction Errors')
    plt.savefig(location + '/z_axis_position_scaled_hist.png')

    plt.figure()
    plt.hist(three_dimension_difference, histtype='step', bins=200, label='Difference', range=(0, 150))
    plt.xlabel('3D Distance to True Vertex (cm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of 3D Prediction Errors')
    plt.savefig(location + '/three_dimension_difference_hist.png')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # readNpzFile(npzFile)
    plot_position_histogram()

