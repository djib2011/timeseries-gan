import numpy as np
import h5py


def load_data_real():

    real_data_train = 'data/yearly_24_nw_train.h5'
    real_data_test = 'data/yearly_24_nw_test.h5'

    with h5py.File(real_data_train, 'r') as hf:
        x_train = np.array(hf.get('X'))
        y_train = np.array(hf.get('y'))

    train = np.c_[x_train, y_train]

    with h5py.File(real_data_test, 'r') as hf:
        x_test = np.array(hf.get('X'))
        y_test = np.array(hf.get('y'))

    test = np.c_[x_test, y_test]

    return train, test


def load_data_fake(samples_file):

    with h5py.File(samples_file, 'r') as hf:
        fake = np.array(hf.get('X'))

    test = fake[:5000]
    train = fake[5000:]

    return train, test


def make_train_test_sets(samples_file):

    r_train, r_test = load_data_real()
    f_train, f_test = load_data_fake(samples_file)

    x_train = np.r_[r_train, f_train]
    y_train = np.array([1] * len(r_train) + [0] * len(f_train))
    ind = np.random.permutation(len(x_train))
    x_train = x_train[ind]
    y_train = y_train[ind]

    x_test = np.r_[r_test, f_test]
    y_test = np.array([1] * len(r_test) + [0] * len(f_test))
    ind = np.random.permutation(len(x_test))
    x_test = x_test[ind]
    y_test = y_test[ind]

    return x_train, x_test, y_train, y_test
