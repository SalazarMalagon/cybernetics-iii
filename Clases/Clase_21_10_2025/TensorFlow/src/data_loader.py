import numpy as np
import scipy.io

def load_data():
    cat_data = scipy.io.loadmat('data/catData_w.mat')
    dog_data = scipy.io.loadmat('data/dogData_w.mat')

    cat_wave = cat_data['cat_wave']
    dog_wave = dog_data['dog_wave']

    x = np.concatenate((dog_wave[:, :40], cat_wave[:, :40]), axis=1)
    x2 = np.concatenate((dog_wave[:, 40:80], cat_wave[:, 40:80]), axis=1)

    labels = np.concatenate((np.ones((40, 1)), np.zeros((40, 1)), np.zeros((40, 1)), np.ones((40, 1))), axis=0).T

    return x, x2, labels, dog_wave, cat_wave