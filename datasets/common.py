import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Union
import tsfresh


def get_last_N(series: Union[pd.Series, np.ndarray], N: int = 18):
    """
    Get the last N points in a timeseries. If len(ts) < N, pad the difference with the first value.
    :param series: A timeseries
    :param N: Number of points to keep
    :return: A timeseries of length N
    """
    ser_N = series.dropna().iloc[-N:].values
    if len(ser_N) < N:
        pad = [ser_N[0]] * (N - len(ser_N))
        ser_N = np.r_[pad, ser_N]
    return ser_N


def load_data(file_pattern: str) -> (np.ndarray,) * 2:
    """
    Load the training and test sets according to a file pattern.

    E.g. if file_pattern = 'data/yearly_24', this function will load:
    'data/yearly_24_train.h5' and 'data/yearly_24_test.h5'

    :param file_pattern: A string that will if followed by '_train.h5' and '_test.h5'
                         contains the training and test sets
    :return: two files, an array containing the train set and an array containing the test set
    """

    real_data_train = file_pattern + '_train.h5'
    real_data_test = file_pattern + '_test.h5'

    with h5py.File(real_data_train, 'r') as hf:
        x_train = np.array(hf.get('X'))
        y_train = np.array(hf.get('y'))

    train = np.c_[x_train, y_train]

    with h5py.File(real_data_test, 'r') as hf:
        x_test = np.array(hf.get('X'))
        y_test = np.array(hf.get('y'))

    test = np.c_[x_test, y_test]

    return train, test


def load_test_set(data_dir: Union[Path, str] = 'data'):

    train_path = Path(data_dir) / 'Yearly-train.csv'
    test_path = Path(data_dir) / 'Yearly-test.csv'

    train_set = pd.read_csv(train_path).drop('V1', axis=1)
    test_set = pd.read_csv(test_path).drop('V1', axis=1)

    X_test = np.array([get_last_N(ser[1], N=18) for ser in train_set.iterrows()])
    y_test = test_set.values

    return X_test, y_test


def normalize_insample(data):
    """
    Normalize by computing the scales the only the insample part of the data

    :param data: a numpy array where each row is a series
    :return: the same array scaled
    """
    mx = data[:, :-6].max(axis=1).reshape(-1, 1)
    mn = data[:, :-6].min(axis=1).reshape(-1, 1)

    if int(mx.max()) == 1 and int(mn.min()) == 0:
        return data

    return (data - mn) / (mx - mn + np.finfo('float').eps)


def normalize_data(data):
    """
    Normalize by computing the scales on all of the series

    :param data: a numpy array where each row is a series
    :return: the same array scaled
    """
    mx = data.max(axis=1).reshape(-1, 1)
    mn = data.min(axis=1).reshape(-1, 1)

    if int(mx.max()) == 1 and int(mn.min()) == 0:
        return data

    return (data - mn) / (mx - mn + np.finfo('float').eps)


def h5_to_npy(h5_file: str):
    """
    Convert an h5 file to a npy file.

    :param h5_file: location of the h5 data
    """

    with h5py.File(h5_file, 'r') as hf:
        data = np.array(hf.get('X'))

    npy_file = h5_file[:-3] + '.npy'

    print('Converting data from "{}" to "{}"'.format(h5_file, npy_file))

    np.save(npy_file, data)


def extract_features(arr: np.ndarray) -> pd.DataFrame:
    """
    Extract tsfresh features from array

    :param data: a numpy array where each row is a series
    :return: DataFrame containing extracted features
    """

    long_df = pd.DataFrame({'ind': [i for series_id in range(arr.shape[0]) for i in [series_id] * arr.shape[1]],
                            'time': list(range(arr.shape[1])) * arr.shape[0],
                            'values': arr.flatten()})

    return tsfresh.extract_features(long_df, column_id='ind', column_sort='time')
