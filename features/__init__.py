from features import extraction
import h5py
import numpy as np


def load_features(pattern):

    with h5py.File(pattern + '_train.h5', 'r') as hf:
        x_train = np.array(hf.get('X'))
        y_train = np.array(hf.get('y'))

    with h5py.File(pattern + '_test.h5', 'r') as hf:
        x_test = np.array(hf.get('X'))
        y_test = np.array(hf.get('y'))

    return np.r_[x_train, x_test], np.r_[y_train, y_test]


def get(name, epoch=None):
    if name == '14k':
        x, y = load_features('features/vanilla_lstm_large_14k/features_epoch_3')
        return x[y == 1]
    elif name == '4k':
        x, y = load_features('features/vanilla_lstm_large_4k/features_epoch_23')
        return x[y == 1]
    else:
        x, y = load_features('features/{}/features_epoch_{}'.format(name, epoch))
        return x[y == 0]
