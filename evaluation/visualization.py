import os
import sys
import argparse
import pickle as pkl
from sklearn.metrics import classification_report

sys.path.append(os.getcwd())

import models
import evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples_file', type=str, help='Path to h5 file containing synthetic samples.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(args.samples_file)

    encoder, autoencoder = models.get_model('autoencoder_2layer_bn')({'base_layer_size': 64, 'input_seq_length': 24})

    autoencoder.fit(x_train, x_train, epochs=args.epochs)

    real = y_test.astype(bool)
    r_2d = encoder.predict(x_test[real])
    f_2d = encoder.predict(x_test[~real])

    # print(r_2d.shape, f_2d.shape)

    with open('/tmp/ae_vis.pkl', 'wb') as f:
        pkl.dump({'real': r_2d, 'fake': f_2d}, f)
