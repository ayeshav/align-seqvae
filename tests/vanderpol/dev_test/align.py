import torch
import torch.nn as nn
from torch.distributions import Normal, Binomial
from utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, K=1, distribution='Normal', linear_flag=True, device='cpu'):
        super().__init__()

        self.dy = dy
        self.dy_ref = dy_ref
        self.distribution = distribution
        self.K = K

        "define the f_enc"
        if linear_flag:
            self.f_enc = nn.Linear(dy, dy_ref).to(device)
        else:
            self.f_enc = nn.Linear(dy, dy_ref).to(device)

        self.f_dec = [nn.Linear(dy_ref, dy).to(device)]

        "include noise scaling for normal distribution"
        if distribution == 'Normal':
            f_dec_var = nn.Linear(dy_ref, dy).to(device)
            self.f_dec.append(f_dec_var)

    def compute_likelihood(self, obsv_params, y):
        "compute likelihood function based on distribution of new dataset"

        if self.distribution == "Normal":
            mu_obsv, var_obsv = obsv_params
            f_dec_mean, f_dec_var = self.f_dec

            mu_obsv_tfm = f_dec_mean(mu_obsv)
            var_obsv_tfm = torch.exp(f_dec_var(var_obsv))
            log_like = torch.sum(Normal(loc=mu_obsv_tfm,
                                        scale=torch.sqrt(var_obsv_tfm)).log_prob(y), (-1, -2))

        elif self.distribution == "Binomial":
            lograte = obsv_params
            f_dec_mean = self.f_dec[0]

            lograte_tfm = f_dec_mean(lograte)

            log_like = torch.sum(Binomial(torch.sigmoid(lograte_tfm)).log_prob(y), (-1,-2))

        return log_like

    def compute_log_prior(self, ref_vae, x):

        log_k_step_prior = 0

        for t in range(x.shape[1] - 1):
            K_ahead = min(self.K, x[:, t + 1:].shape[1])
            _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x[:, t],
                                                                           K_ahead)
            log_k_step_prior = log_k_step_prior + torch.sum(
                Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x[:, t + K_ahead]), -1)

        return log_k_step_prior

    def forward(self, ref_vae, y):

        assert isinstance(ref_vae, SeqVae)

        # apply transformation to data
        y_tfm = self.f_enc(y)  # apply linear transformation to new dataset

        # pass to encoder and get samples
        x_samples = ref_vae.encoder(y_tfm)[0]  # for the given dataset

        log_k_step_prior = self.compute_log_prior(ref_vae, x_samples)

        # get parameters from observation model
        obsv_params = ref_vae.decoder.compute_param(x_samples)

        # transform parameters to new space using decoder
        log_like = self.get_likelihood(obsv_params, y)

        loss = torch.mean(log_like + log_k_step_prior)
        return -loss






