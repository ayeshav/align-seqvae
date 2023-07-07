import math
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from seq_vae import SeqVae
from tqdm import tqdm
from utils import SeqDataLoader


def compute_alignment_loss(ref_vae, f_enc, f_dec, y,
                           latent_mu_ref, latent_cov_ref,
                           beta=1.):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy
    """
    assert isinstance(ref_vae, SeqVae)

    y_tfm = f_enc(y)  # apply linear transformation to new dataset

    x_samples, _, _, _ = ref_vae.encoder(y_tfm)  # for the given dataset

    # compute mean and covariance of latent samples
    latent_mu = torch.mean(torch.mean(x_samples.reshape(-1, y_tfm.shape[-1]), 0, keepdim=True))
    latent_cov = torch.cov(torch.mean(x_samples.reshape(-1, y_tfm.shape[-1]), 0, keepdim=True))

    cov_diff = latent_cov_ref - latent_cov
    reg = torch.sum((latent_mu_ref - latent_mu) ** 2) + torch.trace(cov_diff @ cov_diff.T)

    # measure samples under the log prior to make sure it matches up with the learned generative model
    log_prior = ref_vae._prior(x_samples)

    # we first want to make sure we can reconstruct y_tfm
    mu_like_tfm, var_like_tfm = ref_vae.decoder.compute_param(x_samples)
    y_sample = mu_like_tfm + torch.sqrt(var_like_tfm) * torch.randn(mu_like_tfm.shape, device=mu_like_tfm.device)
    log_like = -0.5 * torch.sum((y - f_dec(y_sample)) ** 2, (-1, -2))
    loss = torch.mean(log_like) - beta * reg
    # TODO: also include log prior in the loss
    return -loss


def train_invertible_mapping(ref_vae, train_dataloader, dy_ref,
                             latent_mu_ref, latent_cov_ref,
                             n_epochs, beta=1., linear_flag=False):
    """
    training function for learning linear alignment and updating prior params
    """
    assert isinstance(train_dataloader, SeqDataLoader)
    assert train_dataloader.shuffle

    dy = train_dataloader.data_tuple[0].shape[-1]
    if linear_flag:
        f_enc = nn.Sequential(*[nn.Linear(dy, 128),
                                nn.ReLU(),
                                nn.Linear(128, dy_ref)]).to(ref_vae.device)
    else:
        f_enc = nn.Linear(dy, dy_ref).to(ref_vae.device)
    f_dec = nn.Linear(dy_ref, dy).to(ref_vae.device)

    training_losses = []
    opt = torch.optim.AdamW(params=list(f_enc.parameters()) + list(f_dec.parameters()),
                            lr=1e-3, weight_decay=1e-4)

    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = compute_alignment_loss(ref_vae, f_enc, f_dec,
                                          y.to(ref_vae.device), beta=beta)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return (f_enc, f_dec), training_losses


