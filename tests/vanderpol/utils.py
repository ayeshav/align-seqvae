import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from seq_vae import SeqVae
from torch.utils.data import Dataset


def compute_elbo(vae, prior, y):

    ll, kl = vae(y, prior)

    elbo = torch.mean(ll - kl)

    return -elbo


def vae_training(vae, prior, epochs, data):

    opt = torch.optim.Adam(params=list(prior.parameters()) + list(vae.parameters()))
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = compute_elbo(vae, prior, y.permute(1, 0, 2))
            loss.backward()
            opt.step()

            with torch.no_grad():
                print(loss.item())
    return vae, prior


class DataSetTs(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.len = y.shape[1]

    def __getitem__(self, idx):
        return self.x[:, idx, :], self.y[:, idx, :]

    def __len__(self):
        return self.len


