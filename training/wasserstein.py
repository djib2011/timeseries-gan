import os
import sys
from pathlib import Path
import pickle as pkl

sys.path.append(os.getcwd())

import datasets
import models

dataset_name = '235k'
name = 'wasserstein_conv_complex'
batch_size = 512
num_blocks = 3

hparams = {'latent_size': 5, 'output_seq_len': 24, 'gp_weight': 10,
           'num_generator_blocks': num_blocks, 'num_critic_blocks': num_blocks}

result_dir = 'results/{}_{}_{}/'.format(name, num_blocks, dataset_name)
report_dir = 'reports/{}_{}_{}/'.format(name, num_blocks, dataset_name)

model_gen = models.get_model('{}_gan'.format(name))

gan = model_gen(hparams)

if dataset_name == '235k':
    train_path = 'data/yearly_24_train.h5'
    test_path = 'data/yearly_24_test.h5'
elif dataset_name == '14k':
    train_path = 'data/yearly_24_nw_train.h5'
    test_path = 'data/yearly_24_nw_test.h5'
else:
    raise ValueError('invalid dataset name')

train_gen = datasets.gan_generator(train_path, batch_size=1024, shuffle=True)
valid_gen = datasets.gan_generator(test_path, batch_size=1024, shuffle=True)

g_losses, d_losses = gan.train(train_gen, valid_gen,
                               train_steps=len(train_gen) // batch_size + 1,
                               valid_steps=len(valid_gen) // batch_size + 1,
                               result_dir=result_dir,
                               save_weights=True,
                               epochs=50)

report_dir = Path(report_dir)

if not report_dir.is_dir():
    os.makedirs(report_dir)

with open(str(report_dir / 'losses.pkl'), 'wb') as f:
    pkl.dump({'generator': g_losses, 'discriminator': d_losses}, f)
