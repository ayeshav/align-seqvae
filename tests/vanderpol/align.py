import math
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from seq_vae import SeqVae
from tqdm import tqdm
from utils import SeqDataLoader


def compute_alignment_loss(ref_vae, params, y, noisy_log_like=False, beta=1.):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy
    """
    assert isinstance(ref_vae, SeqVae)

    f_enc, f_dec = params

    dy, dy_ref = f_enc[:-1].shape  # assumption is that there is no translation
    y_tfm = y @ f_enc[:-1] + f_enc[[-1]]  # apply linear transformation to new dataset

    x_samples, _, _, _ = ref_vae.encoder(y_tfm)  # for the given dataset

    # measure samples under the log prior to make sure it matches up with the learned generative model
    log_prior = ref_vae._prior(x_samples)

    # we first want to make sure we can reconstruct y_tfm
    mu_like_tfm, var_like_tfm = ref_vae.decoder.compute_param(x_samples)
    log_like_tfm = torch.sum(Normal(mu_like_tfm, torch.sqrt(var_like_tfm)).log_prob(y_tfm), (-1, -2))

    elbo = log_like_tfm + log_prior

    if noisy_log_like:
        y_sample = mu_like_tfm + torch.sqrt(var_like_tfm) * torch.randn(mu_like_tfm.shape, device=mu_like_tfm.device)
        log_like = -torch.sum((y - y_sample @ f_dec[:-1] + f_dec[[-1]]) ** 2, (-1, -2))
    else:
        mu_like = mu_like_tfm @ f_dec[:-1] + f_dec[[-1]]
        sigma_like = f_dec[:-1].T @ (var_like_tfm * torch.eye(dy_ref, device=y.device)) @ f_dec[:-1]
        sigma_like = sigma_like + 1e-5 * torch.eye(dy, device=y.device)  # for numerical stability
        log_like = torch.sum(MultivariateNormal(mu_like, covariance_matrix=sigma_like).log_prob(y), -1)

    loss = torch.mean(log_like + beta * elbo)
    # loss = torch.mean(log_like + log_prior - log_q)
    return -loss


def train_invertible_mapping(ref_vae, train_dataloader, dy_ref, n_epochs, noisy_log_like=False, beta=1.):
    """
    training function for learning linear alignment and updating prior params
    """
    dy = train_dataloader.data_tuple[0].shape[-1]
    f_enc = nn.Parameter(torch.randn(dy + 1, dy_ref, device=ref_vae.device) / math.sqrt(dy), requires_grad=True)
    f_dec = nn.Parameter(torch.randn(dy_ref + 1, dy, device=ref_vae.device) / math.sqrt(dy_ref), requires_grad=True)

    training_losses = []
    opt = torch.optim.AdamW(params=[f_enc, f_dec], lr=1e-3, weight_decay=1e-4)

    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = compute_alignment_loss(ref_vae, (f_enc, f_dec), y.to(ref_vae.device),
                                          noisy_log_like=noisy_log_like, beta=beta)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return (f_enc, f_dec), training_losses



