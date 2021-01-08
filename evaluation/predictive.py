import numpy as np
import os
import sys
import argparse
from sklearn import metrics

sys.path.append(os.getcwd())

import models
import evaluation


def split_series(dataset):
    return dataset[:, :-6], dataset[:, -6:]


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

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples_file', type=str, help='Path to h5 file containing synthetic samples.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    args = parser.parse_args()

    real_train, real_test = evaluation.load_data_real()
    fake_train, fake_test = evaluation.load_data_fake(args.samples_file)

    x_real_train, y_real_train = split_series(real_train)
    x_real_test, y_real_test = split_series(real_test)
    x_fake_train, y_fake_train = split_series(fake_train)
    x_fake_test, y_fake_test = split_series(fake_test)

    hparams = {'input_seq_length': 18, 'base_layer_size': 32, 'output_seq_length': 6}
    forecaster_real = models.get_model('forecaster_2layer_bn')(hparams)
    forecaster_fake = models.get_model('forecaster_2layer_bn')(hparams)

    print('Training forecaster on real data')
    forecaster_real.fit(x_real_train, y_real_train, epochs=args.epochs,
                        validation_data=(x_real_test, y_real_test))

    print('Training forecaster on fake data')
    forecaster_fake.fit(x_fake_train, y_fake_train, epochs=args.epochs,
                        validation_data=(x_real_test, y_real_test))

    preds_real = forecaster_real.predict(x_real_test)[..., 0]
    preds_fake = forecaster_fake.predict(x_real_test)[..., 0]

    print('Real targets, real preds:')
    print(forecasting_report(y_real_test, preds_real))
    print('\nReal targets, fake preds:')
    print(forecasting_report(y_real_test, preds_fake))
    print('\nFake targets, real preds:')
    print(forecasting_report(y_fake_test, preds_real))
    print('\nFake targets, fake preds:')
    print(forecasting_report(y_fake_test, preds_fake))
