import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, Bernoulli
from utils import *


def get_likelihood(obsv_params, f_dec, y, distribution='Normal'):
    "compute likelihood function based on distribution of new dataset"

    if distribution == "Normal":
        mu_obsv, var_obsv = obsv_params
        f_dec_mean, f_dec_var = f_dec

        mu_obsv_tfm = f_dec_mean(mu_obsv)
        var_obsv_tfm = torch.exp(f_dec_var(var_obsv))
        log_like = torch.sum(Normal(loc=mu_obsv_tfm,
                                    scale=torch.sqrt(var_obsv_tfm)).log_prob(y), (-1, -2))
    elif distribution == "Poisson" or distribution == "Bernoulli":
        lograte = obsv_params
        f_dec_mean = f_dec[0]

        lograte_tfm = f_dec_mean(lograte)

        log_like = torch.sum(Bernoulli(torch.sigmoid(lograte_tfm)).log_prob(y), (-1,-2))
        # log_like = torch.sum(Poisson(torch.exp(lograte_tfm)).log_prob(y), (-1,-2))

    return log_like


def compute_alignment_loss(ref_vae,
                           f_enc,
                           f_dec,
                           y, K, distribution):
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
    obsv_params = ref_vae.decoder.compute_param(x_samples)

    # transform parameters to new space using decoder
    log_like = get_likelihood(obsv_params, f_dec, y, distribution=distribution)

    loss = torch.mean(log_like + log_k_step_prior)
    return -loss


def get_alignment_params(dy, dy_ref, dy_out=None, distribution='Normal', linear_flag=False, device='cpu'):

    if linear_flag:
        f_enc = Mlp(dy, dy_out, 128).to(device)
        # f_enc = nn.Sequential(*[nn.Linear(dy, 128),
        #                         nn.Softplus(),
        #                         nn.Linear(128, 128),
        #                         nn.Softplus(),
        #                         nn.Linear(128, dy_out)]).to(device)
    else:
        f_enc = nn.Linear(dy, dy_ref).to(device)
        torch.nn.init.normal_(f_enc.weight)

    f_dec = [nn.Linear(dy_ref, dy).to(device)]

    if distribution == 'Normal':

        f_dec_var = nn.Linear(dy_ref, dy).to(device)
        f_dec.append(f_dec_var)

    return f_enc, f_dec


def train_invertible_mapping(ref_vae, train_dataloader, dy_ref,
                             n_epochs,
                             dy_out=None,
                             K=40,
                             lr=1e-3,
                             distribution='Normal',
                             linear_flag=False):
    """
    training function for learning linear alignment and updating prior params
    """
    assert isinstance(train_dataloader, SeqDataLoader)
    assert train_dataloader.shuffle

    dy = train_dataloader.data_tuple[0].shape[-1]
    if dy_out is None:
        dy_out = dy_ref

    f_enc, f_dec = get_alignment_params(dy, dy_ref,dy_out, distribution=distribution,
                                        linear_flag=linear_flag, device=ref_vae.device)
    param_list = list(f_enc.parameters()) + [param for param in nn.ParameterList(f_dec).parameters()]

    training_losses = []
    opt = torch.optim.AdamW(params=param_list,
                            lr=lr, weight_decay=1e-4)
    torch.compile(ref_vae)
    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = compute_alignment_loss(ref_vae, f_enc,
                                          f_dec,
                                          y.to(ref_vae.device),
                                          K=K, distribution=distribution)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(param_list,
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return (f_enc, f_dec), training_losses
