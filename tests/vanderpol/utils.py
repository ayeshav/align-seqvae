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


def obs_alignment(ref_res, prior, y, y_ref, lstq):
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

        u, s, vh = np.linalg.svd(y_ref_cat.T.dot(y_cat).T)

        A = u.dot(vh)
        y_cat_tfm = np.dot(y_cat, A.T) * np.sum(s)

        return (y_cat_tfm@np.linalg.pinv(rp_mat)).reshape(T, N, dy)


class DataSetTs(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.len = y.shape[1]

    def __getitem__(self, idx):
        return self.x[:, idx, :], self.y[:, idx, :]

    def __len__(self):
        return self.len


