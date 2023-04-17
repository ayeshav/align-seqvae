import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from seq_vae import SeqVae


def compute_elbo(vae, prior, y):

    ll, kl = vae(y, prior)

    elbo = torch.mean(ll - kl)

    return -elbo


def vae_training(vae, prior, epochs, data):

    opt = torch.optim.Adam(params=list(prior.parameters()) + list(vae.parameters()))
    for _ in range(epochs):
        opt.zero_grad()
        loss = compute_elbo(vae, prior, data[1])
        loss.backward()
        opt.step()
    return vae
