import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from seq_vae import SeqVae


def compute_elbo(vae, prior, y):

    vae_output = vae(y)

    "compute log prob"
    log_like = torch.sum(Normal(vae_output[0], vae_output[1]).log_prob(y), -1)

    log_prior = torch.sum(Normal(0,1).log_prob(vae_output[2][0]), -1)

    mu, var = prior(vae_output[2][:-1])
    log_prior = log_prior + torch.sum(Normal(mu, var).log_prob(vae_output[2][1:]), 0)

    log_enc = torch.sum(Normal(vae_output[0], vae_output[1]).log_prob(vae_output[2]), -1)

    elbo = torch.mean(log_like - (log_enc - log_prior))

    return -elbo


def vae_training(vae, prior, epochs, data):

    opt = torch.optim.Adam(prior.parameters() + vae.parameters())
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = compute_elbo(vae, prior, y)    # x is of shape batch_size x 1 x 28 x 28
            loss.backward()
            opt.step()
    return vae
