import h5py
import sys
import os

sys.path.append(os.getcwd())

import models

results_dir = 'results/vanilla/'
samples_dir = 'generated/vanilla/'
g_path = results_dir + 'generator_epoch_49.h5'
d_path = results_dir + 'discriminator_epoch_49.h5'
num_samples = 5000
output_len = 24
latent_dim = 5


model = models.get_model('vanilla_gan')({'latent_size': latent_dim, 'output_seq_len': output_len})
model.load_models(g_path, d_path)

synthetic_data = model.generate_n_samples(num_samples)
assert synthetic_data.shape == (num_samples, output_len)

if not os.path.isdir(samples_dir):
    os.makedirs(samples_dir)

with h5py.File(samples_dir + 'predicted.h5', 'w') as hf:
    hf.create_dataset('X', data=synthetic_data.numpy())
