import h5py
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import analysis.utils.binning as bins

def read_fitqun_file(file_path):
    with h5py.File(file_path,mode='r') as h5fw:
        e_1rnll = np.ravel(h5fw['e_1rnll'])
        mu_1rnll = np.ravel(h5fw['mu_1rnll'])
        e_1rmom = np.ravel(h5fw['e_1rmom'])
        labels = np.ravel(h5fw['labels'])
        discr = -e_1rnll + mu_1rnll 
        discr = discr > e_1rmom*0.2
        discr = discr.astype(int)
        temp = np.abs(labels-discr)
        return discr, labels, e_1rmom

def make_fitqunlike_discr(softmax, energies, labels):
    print(softmax)
    discr = softmax[:,1]-softmax[:,0]
    print(discr)
    min = np.argmin(discr)
    max = np.argmax(discr)

    plt.hist2d(energies, softmax[:,1]-softmax[:,0], norm=matplotlib.colors.LogNorm(), cmap=matplotlib.cm.gray)
    plt.colorbar()
    plt.savefig("outputs/2d_softmax_pt_hist.png")
    
    for scale in np.logspace(-4,0,1000):
        temp_discr = discr > energies*scale
        temp_discr = temp_discr.astype(int)
        temp_calc = np.abs(labels-discr)
        temp_metric = np.sum(temp_calc)/len(temp_discr)
        print(f'scale:{scale}, metric: {temp_metric}')