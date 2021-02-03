import os
import sys
from pathlib import Path
import pickle as pkl

sys.path.append(os.getcwd())

import datasets
import models

dataset_name = '14k'

name = 'vanilla_lstm_large_cgan'
batch_size = 512
epochs = 100
result_dir = 'results/{}_{}/'.format(name, dataset_name)
report_dir = 'reports/{}_{}/'.format(name, dataset_name)

model_gen = models.get_model('{}'.format(name))

hparams = {'latent_size': 10, 'output_seq_len': 24, 'condition_size': 10}

gan = model_gen(hparams)

if dataset_name == '235k':
    train_path = 'data/yearly_24_train.h5'
    test_path = 'data/yearly_24_train.h5'
    train_feats_path = 'data/yearly_24_feats_train.h5'
    test_feats_path = 'data/yearly_24_feats_test.h5'
elif dataset_name == '14k':
    train_path = 'data/yearly_24_nw_train.h5'
    test_path = 'data/yearly_24_nw_test.h5'
    train_feats_path = 'data/yearly_24_nw_feats_train.h5'
    test_feats_path = 'data/yearly_24_nw_feats_test.h5'
elif dataset_name == '4k':
    train_path = 'data/yearly_24_undersampled_train.h5'
    test_path = 'data/yearly_24_undersampled_test.h5'
    train_feats_path = 'data/yearly_24_undersampled_feats_train.h5'
    test_feats_path = 'data/yearly_24_undersampled_feats_test.h5'
else:
    raise ValueError('Invalid dataset_name.')

train_gen = datasets.cgan_generator(train_path, train_feats_path, batch_size=1024, shuffle=True)
valid_gen = datasets.cgan_generator(test_path, test_feats_path, batch_size=1024, shuffle=True)

g_losses, d_losses = gan.train(train_gen, valid_gen,
                               train_steps=len(train_gen) // batch_size + 1,
                               valid_steps=len(valid_gen) // batch_size + 1,
                               result_dir=result_dir,
                               save_weights=True,
                               epochs=epochs)

report_dir = Path(report_dir)

if not report_dir.is_dir():
    os.makedirs(report_dir)

with open(str(report_dir / 'losses.pkl'), 'wb') as f:
    pkl.dump({'generator': g_losses, 'discriminator': d_losses}, f)
