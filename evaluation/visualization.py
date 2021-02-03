import numpy as np
import pandas as pd
import os
import sys
import tsfresh
import argparse
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Callable

sys.path.append(os.getcwd())

import models
import evaluation
from datasets import extract_features


def create_autoencoder_visualization(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                     epochs: int = 10) -> (np.ndarray,) * 2:
    """
    Train an autoencoder for projecting the real data vs the generated samples on a 2D space.

    1) Train autoencoder with x_train
    2) Visualize x_test
    3) Partition x_test according to y_test

    :param x_train: array used to train the autoencoder
    :param x_test: array to visualize
    :param y_train: not used (required to wrap function)
    :param y_test: array with 1/0 used to partition x_test into real/fake arrays
    :param epochs: number of epochs to train the Autoencoder
    :return: two arrays containing the coordinates of the real and fake datapoints, respectively
    """

    encoder, autoencoder = models.get_model('autoencoder_2layer_bn')({'base_layer_size': 64,
                                                                      'input_seq_length': x_train.shape[1]})

    autoencoder.fit(x_train, x_train, epochs=epochs)

    real = y_test.astype(bool)
    r_2d = encoder.predict(x_test[real])
    f_2d = encoder.predict(x_test[~real])
    return r_2d, f_2d


def create_pca_visualization(x_train: np.ndarray, x_test: np.ndarray, y_train:np.ndarray,
                             y_test: np.ndarray) -> (np.ndarray, ) * 2:
    """
    PCA for projecting the real data vs the generated samples on a 2D space.

    :param x_train: array used to train fit the PCA
    :param x_test: array to visualize
    :param y_train: not used (required to wrap function)
    :param y_test: array with 1/0 used to partition x_test into real/fake arrays
    :return: two arrays containing the coordinates of the real and fake datapoints, respectively
    """

    pca = PCA(n_components=2)

    pca.fit(x_train)

    real = y_test.astype(bool)
    r_2d = pca.transform(x_test[real])
    f_2d = pca.transform(x_test[~real])

    return r_2d, f_2d


def create_tsne_visualization(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
                              y_test: np.ndarray) -> (np.ndarray, ) * 2:
    """
    t-SNE for projecting the real data vs the generated samples on a 2D space.

    :param x_train: array to be combined with x_test and fit t-SNE
    :param x_test: array to be combined with x_train and fit t-SNE
    :param y_train: not used (required to wrap function)
    :param y_test: array with 1/0 used to partition x_test into real/fake arrays
    :return: two arrays containing the coordinates of the real and fake datapoints, respectively
    """

    tsne = TSNE(n_components=2)

    x = np.r_[x_train, x_test]
    np.random.shuffle(x)
    data_2d = tsne.fit_transform(x)

    real = np.r_[y_train, y_test].astype(bool)

    return data_2d[real], data_2d[~real]


def from_files(func: Callable) -> Callable:
    """
    Wrapper for visualization functions that loads the arrays from files and passes them into the functions.

    :param func: function to wrap
    :return: wrapped function
    """

    def inner(dset_name: str, samples_file: str, *args, **kwargs):
        """
        Visualization function for projecting the real data vs the generated samples on a 2D space.

        Reads the data from files.

        :param samples_file: location of a file containing synthetic samples
        :param dset_name: name of the real dataset
        :return: two arrays containing the coordinates of the real and fake datapoints, respectively
        """
        x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(dset_name, samples_file)
        return func(x_train, x_test, y_train, y_test, *args, **kwargs)

    return inner


def from_features(func: Callable) -> Callable:
    """
    Takes the train/test arrays, extracts features from them, scales them and feeds them to 'func'
    (intended to be one of the visualization functions)

    :param func: function to wrap
    :return: wrapped functions
    """
    def inner(x_train, x_test, y_train, y_test, *args, **kwargs):

        train_feats = extract_features(x_train)
        test_feats = extract_features(x_test)

        na_cols = pd.concat([train_feats, test_feats]).isna().any()

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(train_feats.loc[:, ~na_cols])
        x_test = scaler.transform(test_feats.loc[:, ~na_cols])

        return func(x_train, x_test, y_train, y_test, *args, **kwargs)

    return inner


def from_R_features(func: Callable) -> Callable:
    """
    Takes the patterns of the real and fake feature paths, scales them and feeds them to 'func'
    (intended to be one of the visualization functions)

    :param func: function to wrap
    :return: wrapped functions
    """

    def inner(real_features_pattern, fake_features_pattern):

        real_train = pd.read_csv(real_features_pattern + '_train.csv').values
        real_test = pd.read_csv(real_features_pattern + '_test.csv').values
        fake_train = pd.read_csv(fake_features_pattern + '_train.csv').values
        fake_test = pd.read_csv(fake_features_pattern + '_test.csv').values

        x_train = np.r_[real_train, fake_train]
        y_train = np.array([1] * len(real_train) + [0] * len(fake_train))
        x_test = np.r_[real_test, fake_test]
        y_test = np.array([1] * len(real_test) + [0] * len(fake_test))

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        return func(x_train, x_test, y_train, y_test)

    return inner


create_autoencoder_visualization_from_files = from_files(create_autoencoder_visualization)
create_pca_visualization_from_files = from_files(create_pca_visualization)
create_tsne_visualization_from_files = from_files(create_tsne_visualization)

create_autoencoder_visualization_from_features = from_files(from_features(create_autoencoder_visualization))
create_pca_visualization_from_features = from_files(from_features(create_pca_visualization))
create_tsne_visualization_from_features = from_files(from_features(create_tsne_visualization))

create_autoencoder_visualization_from_R_features = from_R_features(create_autoencoder_visualization)
create_pca_visualization_from_R_features = from_R_features(create_pca_visualization)
create_tsne_visualization_from_R_features = from_R_features(create_tsne_visualization)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Name of the experiment.')
    parser.add_argument('-d', '--dset', type=str, help='Name of the dataset used to train the model.')
    parser.add_argument('-t', '--type', type=str, default='all', help="Type of visualization to run.\n"
                                                                      "Available options = {'all', 'ae', 'pca', 'tsne'}")
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    parser.add_argument('-se', '--samples-epoch', type=int, default=None, help='What epoch to get samples from.')
    args = parser.parse_args()

    samples_dir = 'samples/{}_{}/'.format(args.name, args.dset)
    features_dir = 'features/{}_{}/'.format(args.name, args.dset)
    report_dir = 'reports/{}_{}/'.format(args.name, args.dset)


    if args.samples_epoch is None:
        samples_file = evaluation.find_last_samples_file(samples_dir)
        raise NotImplementedError('Can\'t find last features file.')
    else:
        samples_file = samples_dir + 'samples_epoch_{}.h5'.format(args.samples_epoch)
        features_pattern = features_dir + 'features_epoch_{}'.format(args.samples_epoch)

    print('Getting samples from:', samples_file)

    results = {}

    if args.type.lower() in ('ae', 'all'):
        print('Training Autoencoder for visualization...')
        r_2d, f_2d = create_autoencoder_visualization_from_files(args.dset, samples_file, args.epochs)
        results['ae_raw'] = (r_2d, f_2d)
        r_2d, f_2d = create_autoencoder_visualization_from_features(args.dset, samples_file, args.epochs)
        results['ae_feats'] = (r_2d, f_2d)
        r_2d, f_2d = create_autoencoder_visualization_from_R_features(features_pattern)
        results['ae_R_feats'] = (r_2d, f_2d)
    if args.type.lower() in ('pca', 'all'):
        print('Fitting PCA for visualization...')
        r_2d, f_2d = create_pca_visualization_from_files(args.dset, samples_file)
        results['pca_raw'] = (r_2d, f_2d)
        r_2d, f_2d = create_pca_visualization_from_features(args.dset, samples_file)
        results['pca_feats'] = (r_2d, f_2d)
        r_2d, f_2d = create_pca_visualization_from_R_features(features_pattern)
        results['pca_R_feats'] = (r_2d, f_2d)
    if args.type.lower() in ('tsne', 'all'):
        print('Fitting t-SNE for visualization...')
        r_2d, f_2d = create_tsne_visualization_from_files(args.dset, samples_file)
        results['tsne_raw'] = (r_2d, f_2d)
        r_2d, f_2d = create_tsne_visualization_from_features(args.dset, samples_file)
        results['tsne_feats'] = (r_2d, f_2d)
        r_2d, f_2d = create_tsne_visualization_from_R_features(features_pattern)
        results['tsne_R_feats'] = (r_2d, f_2d)

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    with open(report_dir + '2d_projections_{}.pkl'.format(args.samples_epoch), 'wb') as f:
        pkl.dump(results, f)
