import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from seq_vae import SeqVae, Prior
from torch.utils.data import Dataset


def compute_elbo(vae, prior, y):

    _, ll_params, kl = vae(y, prior)
    elbo = torch.mean(ll_params[2] - kl)

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


def get_predictions(y_aligned, ref_vae, prior):
    assert isinstance(prior, Prior)
    assert isinstance(ref_vae, SeqVae)

    encoder_params, likelihood_params, _ = ref_vae(y_aligned, prior)

    x_samples = encoder_params[2]
    y_recon = likelihood_params[0]


def train_invertible_mapping(epochs, ref_vae, prior, y, y_ref, rp_mat):

    dy = y_ref.shape[2]
    linear_map = nn.Parameter(torch.ones(dy, dy), requires_grad=True)

    opt = torch.optim.Adam(params=linear_map)
    for _ in range(epochs):
        opt.zero_grad()
        loss = compute_map_mse(ref_vae, prior, linear_map, y, rp_mat)
        loss.backward()
        opt.step()

        with torch.no_grad():
            print(loss.item())

    return linear_map


def compute_map_mse(ref_vae, prior, linear_map, y, rp_mat):
    assert isinstance(prior, Prior)
    assert isinstance(ref_vae, SeqVae)

    y_tfm = y@rp_mat@linear_map

    encoder_params, likelihood_params, _ = ref_vae(y_tfm, prior)

    y_tfm_recon = likelihood_params[0]

    "now invert it back to original space"
    y_tfm_recon_original = (y_tfm_recon@np.linalg.pinv(linear_map))@np.linalg.pinv(rp_mat)

    mse = torch.mean((y - y_tfm_recon_original)**2)

    return mse


def obs_alignment(ref_res, prior, y, y_ref, lstq, epochs=20):
    """
    should return an alignment function (can be linear) that takes in y_new
    """
    T, N, dy = y.shape
    dy_ref = y_ref.shape[2]

    if dy != dy_ref:
        rp_mat = np.random.randn(dy, dy_ref)*(1/dy_ref)

    if lstq:

        y_cat, y_ref_cat = y.reshape(-1, dy)@rp_mat, y_ref.reshape(-1, dy_ref)

        y_cat = (y_cat - y_cat.mean(0))/np.linalg.norm(y_cat)
        y_ref_cat = (y_ref_cat - y_ref_cat.mean(0))/np.linalg.norm(y_ref_cat)

        u, s, vh = np.linalg.svd((y_ref_cat.T@y_cat).T)

        A = u.dot(vh)
        y_cat_tfm = (y_cat @ A.T) * np.sum(s)

        # TODO: Figure out how to avoid doing the pinv

        return (y_cat_tfm@np.linalg.pinv(rp_mat)).reshape(T, N, dy)

    else:
        linear_map = train_invertible_mapping(epochs, ref_res, prior, y, y_ref, rp_mat)


class DataSetTs(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.len = y.shape[1]

    def __getitem__(self, idx):
        return self.x[:, idx, :], self.y[:, idx, :]

    def __len__(self):
        return self.len


class Mlp(nn.Module):
    def __init__(self, dx, dy):
        self.linear = nn.Sequential(nn.Linear(dx,dy))

    def forward(self, x):
        return self.linear(x)