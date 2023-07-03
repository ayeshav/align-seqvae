import math
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from seq_vae import SeqVae
from tqdm import tqdm
from utils import SeqDataLoader


def compute_alignment_loss(ref_vae, linear_map, y, noisy_log_like=False):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy
    """
    assert isinstance(ref_vae, SeqVae)

    dy, dy_ref = linear_map.shape  # assumption is that there is no translation
    y_tfm = y @ linear_map  # apply linear transformation to new dataset

    x_samples, _, _, log_q = ref_vae.encoder(y_tfm)  # for the given dataset

    # measure samples under the log prior to make sure it matches up with the learned generative model
    log_prior = ref_vae._prior(x_samples)

    # we first want to make sure we can reconstruct y_tfm
    mu_like_tfm, var_like_tfm = ref_vae.decoder.compute_param(x_samples)
    # log_like_tfm = torch.sum(Normal(mu_like_tfm, torch.sqrt(var_like_tfm)).log_prob(y_tfm), (-1, -2))

    # now let's make sure we can reconstruct the original data
    # let's commit a sin and work with inverses
    inv_linear_map = torch.linalg.pinv(linear_map)

    if noisy_log_like:
        y_sample = mu_like_tfm + torch.sqrt(var_like_tfm) * torch.randn(mu_like_tfm.shape, device=mu_like_tfm.device)
        log_like = torch.sum((y - y_sample) ** 2, (-1, -2))
    else:
        mu_like = mu_like_tfm @ inv_linear_map
        sigma_like = inv_linear_map.T @ (var_like_tfm * torch.eye(dy_ref, device=y.device)) @ inv_linear_map
        sigma_like = sigma_like + 1e-5 * torch.eye(dy, device=y.device)  # for numerical stability
        log_like = torch.sum(MultivariateNormal(mu_like, covariance_matrix=sigma_like).log_prob(y), -1)

    # loss = torch.mean(log_like + log_like_tfm + log_prior - log_q)
    loss = torch.mean(log_like + log_prior - log_q)
    return -loss


def train_invertible_mapping(ref_vae, train_dataloader, dy_ref, n_epochs, noisy_log_like=False):
    """
    training function for learning linear alignment and updating prior params
    """
    dy = train_dataloader.data_tuple[0].shape[-1]
    linear_map = nn.Parameter(torch.randn(dy, dy_ref, device=ref_vae.device) / math.sqrt(dy), requires_grad=True)

    training_losses = []
    opt = torch.optim.AdamW(params=[linear_map], lr=1e-3, weight_decay=1e-4)

    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = compute_alignment_loss(ref_vae, linear_map, y.to(ref_vae.device),
                                          noisy_log_like=noisy_log_like)
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
    dy_ref = y_ref.shape[2]
    #
    # if dy != dy_ref:
    #     rp_mat = torch.randn(dy, dy_ref) * (1 / dy_ref)

    y_dataloader = SeqDataLoader((y,), batch_size=100, shuffle=True)

    linear_map, losses = train_invertible_mapping(ref_vae, y_dataloader, dy_ref, n_epochs)
    return linear_map, losses

