"""
Evaluation scheme is based on TimeGAN:
https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
https://github.com/jsyoon0823/TimeGAN/tree/master/metrics
"""

import os
import sys
sys.path.append(os.getcwd())

from evaluation.common import *
from evaluation import discriminative, predictive, visualization


def to_string(args):
    return ' '.join([str(a) for a in args])


if __name__ == '__main__':
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='Name of the model to load.')
    parser.add_argument('-d', '--dset', type=str, help='Name of the dataset used to train the model.')
    parser.add_argument('-de', '--disc-epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    parser.add_argument('-ae', '--ae-epochs', type=int, default=10, help='Number of epochs to train autoencoder.')
    parser.add_argument('-ge', '--gen-epochs', type=int, default=None, help='Number of epoch of the generator to load.')
    parser.add_argument('-vt', '--vis-type', type=str, default=None, help="Type of visualization to run.\n"
                                                                          "Available options = {'all', 'ae', 'pca', 'tsne'}")
    args = parser.parse_args()

    disc_args = ['python evaluation/discriminative.py', '--name', args.name, '--dset', args.dset, '--epochs', args.disc_epochs]
    os.system(to_string(disc_args))
    pred_args = ['python evaluation/predictive.py', '--name', args.name, '--dset', args.dset, '--epochs', args.disc_epochs]
    os.system(to_string(pred_args))

    vis_args = ['python evaluation/visualization.py', '--name', args.name, '--dset', args.dset]
    if args.gen_epochs is not None:
        vis_args += ['--epoch', args.gen_epochs]
    if args.vis_type:
        vis_args += ['--type', args.vis_type]
    os.system(to_string(vis_args))
