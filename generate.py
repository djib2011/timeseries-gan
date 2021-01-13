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
    parser.add_argument('-e', '--epoch', type=int, default=None, help='Which epoch to load. If not defined, will select'
                                                                      'the last epoch.')

    args = parser.parse_args()

    if args.epoch is None:
        epoch = sorted([int(str(x).split('_')[-1].replace('.h5', ''))
                        for x in Path('results/{}'.format(args.name)).glob('generator*')])[-1]
        print('Epoch not specified. Using last available epoch:', epoch)
    else:
        epoch = args.epoch

    results_dir = 'results/{}/'.format(args.name)
    samples_dir = 'samples/{}/'.format(args.name)
    g_path = results_dir + 'generator_epoch_{}.h5'.format(epoch)
    d_path = results_dir + 'discriminator_epoch_{}.h5'.format(epoch)
    num_samples = 14252
    output_len = 24
    latent_dim = 5


    hparams = {'latent_size': latent_dim, 'output_seq_len': output_len, 'gp_weight': 10}

    model = models.get_model('{}_gan'.format(args.name))(hparams)
    model.load_models(g_path, d_path)

    synthetic_data = model.generate_n_samples(num_samples)
    assert synthetic_data.shape == (num_samples, output_len)

    if not os.path.isdir(samples_dir):
        os.makedirs(samples_dir)

    with h5py.File(samples_dir + 'samples.h5', 'w') as hf:
        hf.create_dataset('X', data=synthetic_data.numpy())

    print('Samples saved at:', samples_dir + 'samples.h5')
