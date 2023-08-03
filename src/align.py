import torch
import torch.nn as nn
from torch.distributions import Normal, Binomial
from utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, f_dec, d_embed=None, K=1, distribution='Normal', linear_flag=True, device='cpu'):
        super().__init__()

        self.dy = dy
        self.dy_ref = dy_ref
        self.distribution = distribution
        self.K = K

        "define the f_enc"
        if linear_flag:
            self.f_enc = nn.Linear(dy, dy_ref).to(device)
        else:
            if d_embed is None:
                raise TypeError("uhoh you forgot to specify d_embed for the featurizer")

            self.f_enc = Mlp(dy, d_embed, 128).to(device)

        self.f_dec = f_dec

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
            K_ahead = min(self.K, x[:, t + 1:].shape[1])
            _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x[:, t],
                                                                           K_ahead)
            log_k_step_prior = log_k_step_prior + torch.sum(
                Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x[:, t + K_ahead]), -1)

        return log_k_step_prior

    def forward(self, ref_vae, y, ref_ss=None):

        # apply transformation to data
        y_tfm = self.f_enc(y)  # apply linear transformation to new dataset

        # pass to encoder and get samples
        x_samples = ref_vae.encoder.sample(y_tfm, align_mode=True)[0]   # for the given dataset

        log_k_step_prior = self.compute_log_prior(ref_vae, x_samples)

        # get likelihood in original space
        log_like = self.f_dec(x_samples, y)

        loss = torch.mean(log_like + log_k_step_prior)

        # optionally compute wasserstein
        # if ref_ss is not None:
        #     mu = torch.mean(x_samples.reshape(x_samples.shape[-1], -1), 1, keepdim=True)
        #     cov = torch.cov(x_samples.reshape(x_samples.shape[-1], -1))
        #     w2 = compute_wasserstein(*ref_ss, mu, cov)
        #     return -loss + w2
        # else:
        return -loss






