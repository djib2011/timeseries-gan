import os
import sys
from pathlib import Path
import pickle as pkl

sys.path.append(os.getcwd())

import datasets
import models


train_path = 'data/yearly_24_nw_train.h5'
test_path = 'data/yearly_24_nw_test.h5'

batch_size = 512
result_dir = 'results/vanilla/'
report_dir = 'reports/vanilla/'

model_gen = models.get_model('vanilla_gan')

hparams = {'latent_size': 5, 'output_seq_len': 24}

gan = model_gen(hparams)

train_gen = datasets.gan_generator(train_path, batch_size=1024, shuffle=True)
valid_gen = datasets.gan_generator(test_path, batch_size=1024, shuffle=True)

g_losses, d_losses = gan.train(train_gen, valid_gen,
                               train_steps=len(train_gen) // batch_size + 1,
                               valid_steps=len(valid_gen) // batch_size + 1,
                               result_dir=result_dir,
                               save_weights=True)

report_dir = Path(report_dir)

if not report_dir.is_dir():
    os.makedirs(report_dir)

with open(str(report_dir / 'losses.pkl'), 'wb') as f:
    pkl.dump({'generator': g_losses, 'discriminator': d_losses}, f)