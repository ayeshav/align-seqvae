import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, f_dec, d_embed=None, linear_flag=True, k_step=1, device='cpu'):
        super().__init__()

        self.dy = dy
        self.dy_ref = dy_ref
        self.k_step = k_step

        "define the f_enc"
        if linear_flag:
            self.f_enc = nn.Linear(dy, dy_ref).to(device)
        else:
            if d_embed is None:
                raise TypeError("uhoh you forgot to specify d_embed for the featurizer")

            self.f_enc = Mlp(dy, d_embed, 64).to(device)

        self.f_dec = f_dec

    # we only need this if we pass through the source decoder first
    # def compute_likelihood(self, decoder, x_samples, y):
    #     "compute likelihood function based on distribution of new dataset"
    #
    #     obsv_params = decoder.compute_param(x_samples)
    #
    #     if self.distribution == "Normal":
    #         mu_obsv, var_obsv = obsv_params
    #
    #         mu_obsv_tfm = self.f_dec_mean(mu_obsv)
    #         var_obsv_tfm = torch.exp(self.f_dec_var(var_obsv))
    #         log_like = torch.sum(Normal(loc=mu_obsv_tfm,
    #                                     scale=torch.sqrt(var_obsv_tfm)).log_prob(y), (-1, -2))
    #
    #     elif self.distribution == "Binomial":
    #         lograte = obsv_params
    #         lograte_tfm = self.f_dec_mean(lograte)
    #
    #         log_like = torch.sum(Binomial(total_count=decoder.total_count,
    #                                       probs=torch.sigmoid(lograte_tfm)).log_prob(y), (-1, -2))
    #
    #     return log_like

    def compute_log_prior(self, ref_vae, x):

        log_k_step_prior = 0

        for t in range(x.shape[1] - 1):
            K_ahead = min(self.k_step, x[:, t + 1:].shape[1])
            _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x[:, t],
                                                                           K_ahead)
            log_k_step_prior = log_k_step_prior + torch.sum(
                Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x[:, t + K_ahead]), -1)

        return log_k_step_prior

    def forward(self, ref_vae, y):

        # apply transformation to data
        y_tfm = self.f_enc(y)  # apply linear transformation to new dataset

        # pass to encoder and get samples
        x_samples, _, _, log_q = ref_vae.encoder(y_tfm)  # for the given dataset

        log_k_step_prior = self.compute_log_prior(ref_vae, x_samples)

        # get likelihood in original space
        log_like = self.f_dec(x_samples, y)

        # get elbo for aligned data
        loss = torch.mean(log_like + log_k_step_prior - log_q)

        return -loss






