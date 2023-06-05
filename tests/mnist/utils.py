import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from vae import VAE


def compute_elbo(vae, x):

    x_c = torch.flatten(x, start_dim=1)

    vae_output = vae(x_c)

    "compute log prob"
    log_like = torch.sum(Bernoulli(vae_output[3]).log_prob(x_c),-1)

    log_prior = torch.sum(Normal(0,1).log_prob(vae_output[2]),-1)

    log_enc = torch.sum(Normal(vae_output[0], vae_output[1]).log_prob(vae_output[2]),-1)

    elbo = torch.mean(log_like - (log_enc - log_prior))

    return -elbo


def vae_training(vae, epochs, data):

    opt = torch.optim.Adam(vae.parameters())
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = compute_elbo(vae, torch.bernoulli(x)) # x is of shape batch_size x 1 x 28 x 28
            loss.backward()
            opt.step()
    return vae


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size())
    x = F.grid_sample(x, grid)
    return x
