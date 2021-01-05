import numpy as np
import tensorflow as tf
import h5py


def seq2seq_generator(data_path: str, batch_size: int = 256, shuffle: bool = True) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.
    Also supports data augmentation and loading series with overlap for backcast.
    :param data_path: Path of a HDF5 file that contains X and y
    :param batch_size: The batch size
    :param shuffle: True/False whether or not the data will be shuffled.
    :return: A TensorFlow data generator.
    """

    # Load data
    with h5py.File(data_path, 'r') as hf:
        x = np.array(hf.get('X'))
        y = np.array(hf.get('y'))

    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(buffer_size=len(x))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data


if __name__ == '__main__':

    data_path = 'data/yearly_24_nw.h5'

    gen = seq2seq_generator(data_path, batch_size=1024, shuffle=True)

    for x, y in gen:
        print('Train set:')
        print(x.shape, y.shape)
        break
