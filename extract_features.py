from sklearn.preprocessing import MinMaxScaler
import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse
import sys

sys.path.append(os.getcwd())

import datasets


def extract_R_features(data: np.ndarray) -> np.ndarray:
    """
    Calls R script that extracts features from several timeseries.

    The R script assumes that the input data can be found in '/tmp/series.csv' and after extracting the features will
    store the new file as '/tmp/output_feats.csv'.

    This function will:
        1. normalize the series
        2. save them as '/tmp/series.csv'
        3. call the R script
        4. load the features from '/tmp/output_feats.csv' and return them

    :param data: a numpy array containing the timeseries
    :return: the extracted features
    """

    data = datasets.normalize_data(data)

    pd.DataFrame(data).to_csv('/tmp/series.csv', header=False, index=False)

    print("Calling 'Spaces.R'")
    os.system('Rscript features/Spaces.R')

    feats = pd.read_csv('/tmp/output_feats.csv')
    return feats.values


def from_real(func):
    """
    Wrapper that calls a func on both the train and test sets of a dataset, after normalizing them
    (intended to be used with 'extract_R_features')

    :param func: a function to wrap
    :return: the wrapped function
    """
    def inner(name):

        train, test = datasets.get(name)

        train_feats = func(train)
        test_feats = func(test)

        return train_feats, test_feats

    return inner


def from_fake(func):
    """
    Wrapper that calls a func on samples from a GAN after first splitting them into a train and a test set
    (intended to be used with 'extract_R_features')

    :param func: a function to wrap
    :return: the wrapped function
    """

    def inner(samples_file, test_size=5000):

        with h5py.File(samples_file, 'r') as hf:
            fake = np.array(hf.get('X'))

        train = fake[:-test_size]
        test = fake[-test_size:]

        train_feats = func(train)
        test_feats = func(test)

        return train_feats, test_feats

    return inner


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', help='Name of the model.')
    parser.add_argument('-d', '--dset', help='Name of the dataset.')
    parser.add_argument('-e', '--epoch', help='Epoch from which the samples were generated.')

    args = parser.parse_args()

    samples_file = 'samples/{}_{}/samples_epoch_{}.h5'.format(args.name, args.dset, args.epoch)
    target_dir = 'features/{}_{}/'.format(args.name, args.dset)

    real_train_feats, real_test_feats = from_real(extract_R_features)(args.dset)
    fake_train_feats, fake_test_feats = from_fake(extract_R_features)(samples_file)

    train_feats = np.r_[real_train_feats, fake_train_feats]
    test_feats = np.r_[real_test_feats, fake_test_feats]

    train_labels = np.array([1] * len(real_train_feats) + [0] * len(fake_train_feats))
    test_labels = np.array([1] * len(real_test_feats) + [0] * len(fake_test_feats))

    if not Path(target_dir).is_dir():
        os.makedirs(target_dir)

    with h5py.File(target_dir + 'features_epoch_{}_train.h5'.format(args.epoch), 'w') as hf:
        hf.create_dataset('X', data=train_feats)
        hf.create_dataset('y', data=train_labels)

    with h5py.File(target_dir + 'features_epoch_{}_test.h5'.format(args.epoch),  'w') as hf:
        hf.create_dataset('X', data=test_feats)
        hf.create_dataset('y', data=test_labels)
