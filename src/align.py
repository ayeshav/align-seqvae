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

    def compute_log_q(self, ref_vae, y):

        x_samples, mu, var = ref_vae.encoder.sample(y)

        if self.k_step == 1:
            log_q = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_samples), (-2, -1))
        else:
            x_k_step = x_samples[:, 1:].unfold(1, self.k_step, 1).permute(0, 1, 3, 2)
            mu_k_step = mu[:, 1:].unfold(1, self.k_step, 1).permute(0, 1, 3, 2)
            var_k_step = var[:, 1:].unfold(1, self.k_step, 1).permute(0, 1, 3, 2)

            log_q = (1/self.k_step) * torch.sum(Normal(mu_k_step, torch.sqrt(var_k_step)).log_prob(x_k_step), (1, 2, 3))

            for i in torch.arange(self.k_step - 1):
                temp = torch.sum(Normal(mu[:, -self.k_step+i+1:], torch.sqrt(var[:, -self.k_step+i+1:])).log_prob(x_samples[:, -self.k_step+i+1:]), (-1,-2))
                log_q = log_q + temp/(self.k_step-1-i)

        return x_samples, mu, var, log_q

    def compute_log_prior(self, ref_vae, x_samples):
        """
        function to compute multi-step prediction (1/K) * sum_{i=1:K} p(x_{t+k} | x_t)
        """

        dx = x_samples.shape[-1]

        if self.k_step == 1:
            log_k_step_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_k_step_prior = log_k_step_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])

        else:
            # get vectorized k-step windows of x_samples BW x k_step x dx
            x_samples_k_step = x_samples[:, 1:].unfold(1, self.k_step, 1).permute(0, 1, 3, 2).reshape(-1, self.k_step, dx)

            _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x_samples[:, :-self.k_step].reshape(-1, dx).unsqueeze(1),
                                                                                   self.k_step, True)

            log_k_step_prior = Normal(torch.hstack(mu_k_ahead), torch.sqrt(torch.vstack(var_k_ahead))).log_prob(x_samples_k_step)
            log_k_step_prior = torch.sum(log_k_step_prior.reshape(x_samples.shape[0], -1, self.k_step, dx),(1, 2, 3))/self.k_step

            #TODO: can we remove this loop?
            for i in torch.arange(self.k_step - 1):
                K_ahead = min(self.k_step, x_samples[:, -self.k_step+i+1:].shape[1])

                _, mu_k, var_k = ref_vae.prior.sample_k_step_ahead(x_samples[:, -self.k_step + i].reshape(-1, dx).unsqueeze(1), K_ahead, True)

                log_prior = torch.sum(Normal(torch.hstack(mu_k), torch.sqrt(torch.vstack(var_k))).log_prob(
                                x_samples[:, -self.k_step + i + 1:].reshape(-1, K_ahead, dx)), (-1, -2))

                log_k_step_prior = log_k_step_prior + log_prior/K_ahead

        return log_k_step_prior

    def forward(self, ref_vae, y, beta=1.0):

        # apply transformation to data
        y_tfm = self.f_enc(y)  # apply linear transformation to new dataset

        # pass to encoder and get samples
        x_samples, _, _, log_q = self.compute_log_q(ref_vae, y_tfm)

        log_k_step_prior = self.compute_log_prior(ref_vae, x_samples)

        # get likelihood in original space
        log_like = self.f_dec(x_samples, y)

        # get elbo for aligned data
        loss = torch.mean(log_like + beta * (log_k_step_prior - log_q))

        return -loss






