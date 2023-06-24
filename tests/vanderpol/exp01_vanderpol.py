import os

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
data = torch.load('data/noisy_vanderpol.pt')


def train_ref_vae():
    "extract reference data"
    x, y = data[0]['x'], data[0]['y']
    x_train, y_train = x[:, :n_train, :], y[:, :n_train, :]  # Time by Batch by Dimension

    data_ref = SeqDataLoader((x_train.float(), y_train.float()), batch_size=100)

    dy_ref = y.shape[2]

    prior = Prior(dx)
    vae = SeqVae(dx, dy_ref, dh)
    res = vae_training(vae, prior, 300, data_ref)

    torch.save(res, 'trained_models/reference_model.pt')


def reuse_dynamics(reference, epochs=20):

    vae, prior, _ = torch.load(reference)

    # for i in range(1, len(data)):
    res_alignment = obs_alignment(vae, prior, data[1]['y'].float(), data[0]['y'].float(), epochs=epochs)

    return res_alignment

def main():

    model_path = 'trained_models'
    results_path = 'results'

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if not os.path.isfile(model_path + 'reference_model.pt'):
        train_ref_vae()

    res_alignment = reuse_dynamics(model_path + 'reference_model.pt', 50)
    #
    # torch.save(res_alignment, 'result.pt')


if __name__ == '__main__':
    main()






