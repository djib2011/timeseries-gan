import h5py
import sys
import os

sys.path.append(os.getcwd())

import models

model_name = 'vanilla_lstm_large'
epoch = 49
results_dir = 'results/{}/'.format(model_name)
samples_dir = 'synthetic/{}/'.format(model_name)
g_path = results_dir + 'generator_epoch_{}.h5'.format(epoch)
d_path = results_dir + 'discriminator_epoch_{}.h5'.format(epoch)
num_samples = 14252
output_len = 24
latent_dim = 5


model = models.get_model('{}_gan'.format(model_name))({'latent_size': latent_dim, 'output_seq_len': output_len})
model.load_models(g_path, d_path)

synthetic_data = model.generate_n_samples(num_samples)
assert synthetic_data.shape == (num_samples, output_len)

if not os.path.isdir(samples_dir):
    os.makedirs(samples_dir)

with h5py.File(samples_dir + 'predicted.h5', 'w') as hf:
    hf.create_dataset('X', data=synthetic_data.numpy())

