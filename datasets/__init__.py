from datasets.common import *
from datasets.generators import seq2seq_generator, gan_generator, cgan_generator


def get(name):
    if name == '235k':
        return load_data('data/yearly_24')
    elif name == '14k':
        return load_data('data/yearly_24_nw')
    elif name == '4k':
        return load_data('data/yearly_24_undersampled')
    else:
        raise ValueError('Invalid dataset name.')