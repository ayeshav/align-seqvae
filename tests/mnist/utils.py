import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from vae import VAE


def compute_elbo(vae, x):

    x_c = torch.flatten(x, start_dim=1)

    vae_output = vae(x_c)

    "compute log prob"
    log_like = Bernoulli(vae_output[3]).log_prob(x_c)

    log_prior = Normal(0,1).log_prob(vae_output[2])

    log_enc = Normal(vae_output[0], vae_output[1]).log_prob(vae_output[2])

    elbo = log_like + log_enc - log_prior

    return -elbo


def vae_training(vae, epochs, data):

    opt = torch.optim.Adam(vae.parameters())
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = compute_elbo(vae, x) # x is of shape batch_size x 1 x 28 x 28
            loss.backward()
            opt.step()

    return vae






