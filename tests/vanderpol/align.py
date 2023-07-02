import math
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from seq_vae import SeqVae
from tqdm import tqdm


def compute_map_mse(ref_vae, linear_map, y):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    prior: pre-trained prior
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy TODO: shouldn't it be Time by K by dy?
    """
    assert isinstance(ref_vae, SeqVae)

    dy, dy_ref = linear_map.shape  # Assumption right now is that dy and dy_ref are of same dimension and no translations
    y_tfm = y @ linear_map  # apply linear transformation to new dataset

    x_samples, _, _, _ = ref_vae.encoder(y_tfm)  # for the given dataset

    # measure samples under the log prior to make sure it matches up with the learned generative model
    log_prior = ref_vae._prior(x_samples)

    # now, we want to make sure we can reconstruct the original data, NOT y_tfm
    # TODO: can we show that reconstructing y_tfm is equivalent to learning to reconstruct y???
    mu_like_tfm, sigma_like_tfm = ref_vae.decoder.compute_param(x_samples)

    # let's commit a sin and work with inverses
    inv_linear_map = torch.linalg.pinv(linear_map)
    mu_like = mu_like_tfm @ inv_linear_map
    sigma_like = inv_linear_map.T @ (sigma_like_tfm * torch.eye(dy_ref)) @ inv_linear_map
    sigma_like = sigma_like + 1e-5 * torch.eye(dy)  # for numerical stability
    log_like = torch.sum(MultivariateNormal(mu_like, covariance_matrix=sigma_like).log_prob(y))

    loss = torch.mean(log_like + log_prior)
    return -loss


def train_invertible_mapping(ref_vae, y, dy_ref, n_epochs):
    """
    training function for learning linear alignment and updating prior params
    """
    dy = y.shape[2]
    linear_map = nn.Parameter(torch.randn(dy, dy_ref) / math.sqrt(dy), requires_grad=True)

    training_losses = []
    opt = torch.optim.Adam(params=[linear_map], lr=1e-3)

    for _ in tqdm(range(n_epochs)):

        # for y_b,  in data:
        opt.zero_grad()
        loss = compute_map_mse(ref_vae, linear_map, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            training_losses.append(loss.item())

    return linear_map, training_losses


def obs_alignment(ref_vae, y, y_ref, n_epochs=20):
    """
    ref_res: reference vae trained on y_ref
    prior: trained prior on y_ref
    y: new data to be aligned of shape K x T x dy
    y_ref: reference dataset of shape K x T x dy_ref

    returns linear map, rp_mat and trained prior
    """
    T, N, dy = y.shape
    dy_ref = y_ref.shape[2]

    if dy != dy_ref:
        rp_mat = torch.randn(dy, dy_ref) * (1 / dy_ref)

    linear_map, prior = train_invertible_mapping(ref_vae, y, dy_ref, n_epochs)
    return linear_map, rp_mat, prior

