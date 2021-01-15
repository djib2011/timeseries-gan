import os
import sys
from sklearn.metrics import classification_report

sys.path.append(os.getcwd())

import models
import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Name of the model to load.')
    parser.add_argument('-d', '--dset', type=str, help='Name of the dataset used to train the model.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    args = parser.parse_args()

    samples_file = 'samples/{}_{}/samples.h5'.format(args.name, args.dset)
    print('Getting samples from:', samples_file)

    report_dir = 'reports/{}_{}/'.format(args.name, args.dset)

    x_train, x_test, y_train, y_test = evaluation.make_train_test_sets(samples_file)

    discriminator = models.get_model('classifier_3layer_bn')({'base_layer_size': 64, 'input_seq_length': 24})

    discriminator.fit(x_train, y_train, epochs=args.epochs)

    y_hat = discriminator.predict(x_test)

    out = 'Discriminative Report:\n' + classification_report(y_test, y_hat)
    print(out)

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    with open(report_dir + 'predictive_evaluation.txt', 'w') as f:
        f.write(out)
