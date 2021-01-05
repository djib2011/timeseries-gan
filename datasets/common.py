import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


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


def load_test_set(data_dir: Union[Path, str] = 'data'):

    train_path = Path(data_dir) / 'Yearly-train.csv'
    test_path = Path(data_dir) / 'Yearly-test.csv'

    train_set = pd.read_csv(train_path).drop('V1', axis=1)
    test_set = pd.read_csv(test_path).drop('V1', axis=1)

    X_test = np.array([get_last_N(ser[1], N=18) for ser in train_set.iterrows()])
    y_test = test_set.values

    return X_test, y_test


def normalize_data(data):

    mx = data[:, :-6].max(axis=1).reshape(-1, 1)
    mn = data[:, :-6].min(axis=1).reshape(-1, 1)

    return (data - mn) / (mx - mn + np.finfo('float').eps)