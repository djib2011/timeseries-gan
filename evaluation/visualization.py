import numpy as np
import os
import sys
import argparse
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.getcwd())

import models
import evaluation


def create_autoencoder_visualization(samples_file, epochs):

    x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(samples_file)

    encoder, autoencoder = models.get_model('autoencoder_2layer_bn')({'base_layer_size': 64, 'input_seq_length': 24})

    autoencoder.fit(x_train, x_train, epochs=epochs)

    real = y_test.astype(bool)
    r_2d = encoder.predict(x_test[real])
    f_2d = encoder.predict(x_test[~real])
    return r_2d, f_2d


def create_pca_visualization(samples_file):
    x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(samples_file)

    pca = PCA(n_components=2)

    pca.fit(x_train)

    real = y_test.astype(bool)
    r_2d = pca.transform(x_test[real])
    f_2d = pca.transform(x_test[~real])

    return r_2d, f_2d


def create_tsne_visualization(samples_file):
    x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(samples_file)

    tsne = TSNE(n_components=2)

    data_2d = tsne.fit_transform(np.r_[x_train, x_test])

    real = np.r_[y_train, y_test].astype(bool)

    return data_2d[real], data_2d[~real]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Name of the experiment.')
    parser.add_argument('-t', '--type', type=str, default='all', help="Type of visualization to run.\n"
                                                                      "Available options = {'all', 'ae', 'pca', 'tsne'}")
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    args = parser.parse_args()

    result_dir = 'reports/{}/'.format(args.name)
    samples_file = 'samples/{}/samples.h5'.format(args.name)
    print('Getting samples from:', samples_file)

    results = {}

    if args.type.lower() in ('ae', 'all'):
        print('Training Autoencoder for visualization...')
        r_2d, f_2d = create_autoencoder_visualization(samples_file, args.epochs)
        results['ae'] = (r_2d, f_2d)
    if args.type.lower() in ('pca', 'all'):
        print('Fitting PCA for visualization...')
        r_2d, f_2d = create_pca_visualization(samples_file)
        results['pca'] = (r_2d, f_2d)
    if args.type.lower() in ('tsne', 'all'):
        print('Fitting t-SNE for visualization...')
        r_2d, f_2d = create_tsne_visualization(samples_file)
        results['tsne'] = (r_2d, f_2d)


    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + '2d_projections.pkl', 'wb') as f:
        pkl.dump(results, f)

