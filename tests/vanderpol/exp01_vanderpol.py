import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *
from seq_vae import SeqVae, Prior

import matplotlib.pyplot as plt

dx = 2
dh = 256

n_train = 300

"load vanderpol data"
data = torch.load('vanderpol.pt')


def train_ref_vae():
    "extract reference data"
    x, y = data['0']
    x_train, y_train = x[:, :n_train, :], y[:, :n_train, :]

    data_ref = DataLoader(DataSetTs(x_train.float(), y_train.float()), batch_size=80)

    dy_ref = y.shape[2]

    prior = Prior(dx)
    vae = SeqVae(dx, dy_ref, dh)
    res = vae_training(vae, prior, 100, data_ref)

    torch.save(res, 'reference_model.pt')


def reuse_dynamics(reference, lstq):

    vae = torch.load(reference)

    for key in data:
        res_alignment = obs_alignment(vae, data[key], lstq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--train_prior', help='1 to specify if the vae needs to be trained from scratch')
    parser.add_argument('-l', '--lstq', type=int, help='1 if lstq can be used for alignment')
    parser.add_argument('f', '--path', type=str, help='if train_prior is set to 0, provide path of trained model')

    args = parser.parse_args()

    if args.train_prior == 1:
        train_ref_vae()
        fname = 'reference_model.pt'
    else:
        fname = args.path

    reuse_dynamics(fname, bool(args.lstq))


if __name__ == '__main__':
    main()






