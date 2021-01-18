import numpy as np
from pathlib import Path
import h5py

import os
import sys
sys.path.append(os.getcwd())

import datasets


def load_data_real(dset_name: str) -> (np.ndarray,) * 2:
    """
    Load file containing real series.

    :param dset_name: name of the dataset
    :return: real data split into training and test sets
    """

    return datasets.get(dset_name)


def load_data_fake(samples_file: str) -> (np.ndarray,) * 2:
    """
    Load file containing fake series.

    :param samples_file: location of a file containing synthetic samples
    :return: real data split into training and test sets
    """

    with h5py.File(samples_file, 'r') as hf:
        fake = np.array(hf.get('X'))

    test = fake[:5000]
    train = fake[5000:]

    return train, test


def make_train_test_sets(dset_name: str, samples_file: str) -> (np.ndarray,) * 4:
    """
    Make train and test sets for discriminating between real and fake series

    :param dset_name: name of the real dataset
    :param samples_file: location of a file containing synthetic samples
    :return: X_train, X_test, y_train, y_test
    """
    r_train, r_test = load_data_real(dset_name)
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


def find_last_samples_file(samples_dir: str) -> str:
    """
    Find the samples from the samples_dir that correspond to the last epoch

    :param samples_dir: directory containing samples files
    :return: the path to the latest samples dir
    """
    return sorted([str(s) for s in Path(samples_dir).glob('*.h5')])[-1]
