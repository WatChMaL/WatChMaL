import h5py
import numpy as np

import matplotlib.pyplot as plt
import math
from numpy.core.defchararray import index
plt.style.use('seaborn-whitegrid')


def 
X_POS = 0
Y_POS = 1
Z_POS = 2

electron_path = '/home/ykwon/electron.npz'
data_path = "/fast_scratch/WatChMaL/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emg_E0to1000MeV_digihits.h5"

predicted_path = [
  "/home/jyang/project/regression-project/outputs/2021-11-03/12-47-21/outputs/predictions.npy", # z-pos, no flips
]

elec_test = np.load(electron_path)['test_idxs']
preds_elec = np.load(predicted_path[0])

data = h5py.File(data_path, 'r')
energy_data = data["energies"]
position_data = data["positions"]

def compare(actual, predicted):
  return (predicted - actual) / actual

x = [position_data[elec_test[index]][0][X_POS] for index in range(len(preds_elec))]
y = [position_data[elec_test[index]][0][Y_POS] for index in range(len(preds_elec))]
z = [position_data[elec_test[index]][0][Z_POS] for index in range(len(preds_elec))]


transverse = np.sqrt(np.power(x, 2) + np.power(z, 2))
theta = [math.atan(z[i] / x[i]) for i in range(len(z))]

#print(energy_data[elec_test[0]])
plt.scatter(x, y, color='black', s=3)
plt.savefig('./y-pos-horizontal.png')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # readNpzFile(npzFile)
    plot_energy_histogram()
