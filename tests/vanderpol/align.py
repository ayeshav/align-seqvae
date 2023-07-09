import torch.nn as nn
from torch.distributions import Normal
from utils import *


def compute_alignment_loss(ref_vae,
                           f_enc,
                           f_dec_mean, f_dec_var,
                           y, K):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy
    """
    assert isinstance(ref_vae, SeqVae)

    # apply transformation to data
    y_tfm = f_enc(y)  # apply linear transformation to new dataset

    # pass to encoder and get samples
    x_samples, _, _, _ = ref_vae.encoder(y_tfm)  # for the given dataset

    log_k_step_prior = 0

    for t in range(x_samples.shape[1] - 1):
        K_ahead = min(K, x_samples[:, t + 1:].shape[1])
        _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x_samples[:, t],
                                                                       K_ahead)
        log_k_step_prior = log_k_step_prior + torch.sum(Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)

    # get parameters from observation model
    mu_obsv, var_obsv = ref_vae.decoder.compute_param(x_samples)

    # transform parameters to new space using decoder
    mu_obsv_tfm = f_dec_mean(mu_obsv)
    var_obsv_tfm = torch.exp(f_dec_var(var_obsv))
    log_like = torch.sum(Normal(loc=mu_obsv_tfm,
                                scale=torch.sqrt(var_obsv_tfm)).log_prob(y), (-1, -2))
    loss = torch.mean(log_like + log_k_step_prior)
    return -loss


def train_invertible_mapping(ref_vae, train_dataloader, dy_ref,
                             n_epochs,
                             K=40,
                             linear_flag=False):
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
    f_dec_mean = nn.Linear(dy_ref, dy).to(ref_vae.device)
    f_dec_var = nn.Linear(dy_ref, dy).to(ref_vae.device)

    training_losses = []
    opt = torch.optim.AdamW(params=list(f_enc.parameters()) + list(f_dec_mean.parameters()) + list(f_dec_var.parameters()),
                            lr=5e-4, weight_decay=1e-4)
    torch.compile(ref_vae)
    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = compute_alignment_loss(ref_vae, f_enc,
                                          f_dec_mean, f_dec_var,
                                          y.to(ref_vae.device),
                                          K=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(f_enc.parameters()) + list(f_dec_mean.parameters()) + list(f_dec_var.parameters()),
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return (f_enc, f_dec_mean, f_dec_var), training_losses
