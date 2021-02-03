import numpy as np
import h5py
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

import models

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', type=str, help='Name of the model to load.')
    parser.add_argument('-d', '--dset', type=str, help='Name of the dataset used to train the model.')
    parser.add_argument('-e', '--epoch', type=int, default=None, help='Which epoch to load. If not defined, will select'
                                                                      'the last epoch.')

    args = parser.parse_args()

    if args.dset == '14k':
        train_feats_path = 'data/yearly_24_nw_feats_train.h5'
        test_feats_path = 'data/yearly_24_nw_feats_test.h5'
    else:
        raise NotImplementedError('Only 14k dataset supported for cGAN')

    with h5py.File(train_feats_path, 'r') as hf:
        train_feats = np.array(hf.get('X'))

    with h5py.File(test_feats_path, 'r') as hf:
        test_feats = np.array(hf.get('X'))

    feats = np.r_[train_feats, test_feats]

    results_dir = 'results/{}_{}/'.format(args.name, args.dset)
    samples_dir = 'samples/{}_{}/'.format(args.name, args.dset)

    if args.epoch is None:
        epoch = sorted([int(str(x).split('_')[-1].replace('.h5', ''))
                        for x in Path(results_dir).glob('generator*')])[-1]
        print('Epoch not specified. Using last available epoch:', epoch)
    else:
        epoch = args.epoch

    g_path = results_dir + 'generator_epoch_{}.h5'.format(epoch)
    d_path = results_dir + 'discriminator_epoch_{}.h5'.format(epoch)
    output_len = 24
    latent_dim = feats.shape[1]

    hparams = {'latent_size': latent_dim, 'output_seq_len': output_len, 'gp_weight': 10, 'condition_size': feats.shape[1]}

    model = models.get_model('{}'.format(args.name))(hparams)
    model.load_models(g_path, d_path)

    synthetic_data = model.generate_n_samples(feats)
    assert synthetic_data.shape == (feats.shape[0], output_len)

    if not os.path.isdir(samples_dir):
        os.makedirs(samples_dir)

    with h5py.File(samples_dir + 'samples_epoch_{}.h5'.format(epoch), 'w') as hf:
        hf.create_dataset('X', data=synthetic_data.numpy())

    print('Samples saved at:', samples_dir + 'samples_epoch_{}.h5'.format(epoch))
