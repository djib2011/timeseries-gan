import os
import sys
import argparse
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

    discriminator = models.get_model('classifier_3layer_bn')({'base_layer_size': 64, 'input_seq_length': 24})

    discriminator.fit(x_train, y_train, epochs=args.epochs)

    y_hat = discriminator.predict(x_test)

    print(classification_report(y_test, y_hat))
