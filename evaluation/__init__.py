"""
Evaluation scheme is based on TimeGAN:
https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
https://github.com/jsyoon0823/TimeGAN/tree/master/metrics
"""
from evaluation.common import *
from evaluation import discriminative, predictive, visualization

if __name__ == '__main__':
    pass
    # import argparse
    # import subprocess
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--name', type=str, help='Name of the model to load.')
    # parser.add_argument('-de', '--disc-epochs', type=int, default=10, help='Number of epochs to train discriminator.')
    # parser.add_argument('-ae', '--ae-epochs', type=int, default=10, help='Number of epochs to train autoencoder.')
    # parser.add_argument('-ge', '--gen-epochs', type=int, default=None, help='Number of epoch of the generator to load.')
    # parser.add_argument('-vt', '--vis-type', type=str, default=None, help="Type of visualization to run.\n"
    #                                                                       "Available options = {'all', 'ae', 'pca', 'tsne'}")
    # args = parser.parse_args()
    #
    # subprocess.run('evaluation/discriminative.py', ['--name', args.name, '--epochs', args.de])
    # subprocess.run('evaluation/predictive.py', ['--name', args.name, '--epochs', args.de])
    #
    # vis_args = ['--name', args.name]
    # if args.ge is not None:
    #     vis_args += ['--epoch', args.ge]
    # if args.vt:
    #     vis_args += ['--type', args.vt]
    # subprocess.run('evaluation/visualization.py', vis_args)
