import numpy as np
import os
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

import models
import evaluation


def split_series(dataset):
    return dataset[:, :-6], dataset[:, -6:]


def combine_datasets(x_dsets, y_dsets):
    combined_x = np.concatenate(x_dsets)
    combined_y = np.concatenate(y_dsets)
    ind = np.random.permutation(len(x_dsets))
    return combined_x[ind], combined_y[ind]


def SMAPE(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Computes the symmetric Mean Absolute Percentage Error (sMAPE) amongst the targets (y) and the insample (x)
    :param y: Array containing out-of-sample data poitns (i.e. targets). Should be (num_samples, forecast_horizon)
    :param p: Array containing predictions. Should be (num_samples, forecast_horizon)
    :return: The sMAPE per sample (num_samples,)
    """
    nom = np.abs(y - p)
    denom = np.abs(y) + np.abs(p) + np.finfo('float').eps
    return 2 * np.mean(nom / denom, axis=1) * 100


def MAPE(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100


def forecasting_report(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean(np.abs((y_true - y_pred) ** 2))
    mape = np.mean(MAPE(y_true, y_pred))
    smape = np.mean(SMAPE(y_true, y_pred))
    return 'MAE:   {:.3f}\nMSE:   {:.3f}\nMAPE:  {:.3f}\nsMAPE: {:.3f}'.format(mae, mse, mape, smape)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Name of the model to load.')
    parser.add_argument('-d', '--dset', type=str, help='Name of the dataset used to train the model.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    parser.add_argument('-se', '--samples-epoch', type=int, default=None, help='What epoch to get samples from.')
    args = parser.parse_args()

    print(dir(args))
    samples_dir = 'samples/{}_{}/'.format(args.name, args.dset)
    if args.samples_epoch is None:
        samples_file = evaluation.find_last_samples_file(samples_dir)
    else:
        samples_file = samples_dir + 'samples_epoch_{}.h5'.format(args.samples_epoch)

    print('Getting samples from:', samples_file)

    report_dir = 'reports/{}_{}/'.format(args.name, args.dset)

    real_train, real_test = evaluation.load_data_real()
    fake_train, fake_test = evaluation.load_data_fake(samples_file)

    x_real_train, y_real_train = split_series(real_train)
    x_real_test, y_real_test = split_series(real_test)
    x_fake_train, y_fake_train = split_series(fake_train)
    x_fake_test, y_fake_test = split_series(fake_test)

    x_comb_train, y_comb_train = combine_datasets((x_real_train, x_fake_train), (y_real_train, y_fake_train))

    hparams = {'input_seq_length': 18, 'base_layer_size': 32, 'output_seq_length': 6}
    forecaster_real = models.get_model('forecaster_2layer_bn')(hparams)
    forecaster_fake = models.get_model('forecaster_2layer_bn')(hparams)
    forecaster_comb = models.get_model('forecaster_2layer_bn')(hparams)

    print('Training forecaster on real data')
    forecaster_real.fit(x_real_train, y_real_train, epochs=args.epochs,
                        validation_data=(x_real_test, y_real_test))

    print('Training forecaster on fake data')
    forecaster_fake.fit(x_fake_train, y_fake_train, epochs=args.epochs,
                        validation_data=(x_real_test, y_real_test))

    print('Training forecaster on combined data')
    forecaster_comb.fit(x_comb_train, y_comb_train, epochs=args.epochs,
                        validation_data=(x_real_test, y_real_test))

    preds_real = forecaster_real.predict(x_real_test)[..., 0]
    preds_fake = forecaster_fake.predict(x_real_test)[..., 0]
    preds_comb = forecaster_comb.predict(x_real_test)[..., 0]

    out = 'Real targets, real training set:\n'
    out += forecasting_report(y_real_test, preds_real)
    out += '\n\nReal targets, fake training set:\n'
    out += forecasting_report(y_real_test, preds_fake)
    # out += '\n\nFake targets, real training set:\n'
    # out += forecasting_report(y_fake_test, preds_real)
    # out += '\n\nFake targets, fake training set:\n'
    # out += forecasting_report(y_fake_test, preds_fake)
    out += '\n\nReal targets, combined training set:\n'
    out += forecasting_report(y_real_test, preds_comb)

    print(out)

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    with open(report_dir + 'predictive_evaluation.txt', 'w') as f:
        f.write(out)
