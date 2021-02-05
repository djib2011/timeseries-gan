import numpy as np
import h5py

import os
import sys
sys.path.append(os.getcwd())

from features.extraction import *

name = 'vanilla_lstm_large_cgan_2'
dset = '14k'

real_train_feats, real_test_feats = from_real(extract_R_features)(dset)

target_dir = 'features/{}_{}/'.format(name, dset)

if not Path(target_dir).is_dir():
    os.makedirs(target_dir)

for epoch in range(50):

    samples_file = 'samples/{}_{}/samples_epoch_{}.h5'.format(name, dset, epoch)

    fake_train_feats, fake_test_feats = from_fake(extract_R_features)(samples_file)

    train_feats = np.r_[real_train_feats, fake_train_feats]
    test_feats = np.r_[real_test_feats, fake_test_feats]

    train_labels = np.array([1] * len(real_train_feats) + [0] * len(fake_train_feats))
    test_labels = np.array([1] * len(real_test_feats) + [0] * len(fake_test_feats))

    with h5py.File(target_dir + 'features_epoch_{}_train.h5'.format(epoch), 'w') as hf:
        hf.create_dataset('X', data=train_feats)
        hf.create_dataset('y', data=train_labels)

    with h5py.File(target_dir + 'features_epoch_{}_test.h5'.format(epoch), 'w') as hf:
        hf.create_dataset('X', data=test_feats)
        hf.create_dataset('y', data=test_labels)
