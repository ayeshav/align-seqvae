import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, f_dec, readout_behav=None, d_embed=None, du=0,
                 linear_flag=True, k_step=1, device='cpu'):
        super().__init__()

        self.dy = dy
        self.dy_ref = dy_ref
        self.du = du

        self.k_step = k_step

        "define the f_enc"
        if linear_flag:
            self.f_enc = nn.Linear(dy, dy_ref).to(device)
        else:
            if d_embed is None:
                raise TypeError("uhoh you forgot to specify d_embed for the featurizer")

            self.f_enc = Mlp(dy + du, d_embed, 64).to(device)

        self.f_dec = f_dec
        self.readout_behav = readout_behav

    def compute_log_q(self, ref_vae, y):

        x_samples, mu, var = ref_vae.encoder.sample(y)

        if self.k_step == 1:
            log_q = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_samples), (-2, -1))
        else:
            x_k = vectorize_x(x_samples[:, 1:], self.k_step)
            mu_k = vectorize_x(mu[:, 1:], self.k_step)
            var_k_step = vectorize_x(var[:, 1:], self.k_step)

            "mean across k steps"
            log_q = torch.mean(Normal(mu_k, torch.sqrt(var_k_step)).log_prob(x_k), -2)

            "sum across time and dx"
            log_q = torch.sum(log_q.reshape(mu.shape[0], -1, mu.shape[2]), (-1, -2))

            for i in torch.arange(self.k_step - 1):
                K_ahead = self.k_step - i - 1
                temp = torch.sum(Normal(mu[..., -K_ahead:, :], torch.sqrt(var)[..., -K_ahead:, :]).log_prob(
                    x_samples[..., -K_ahead:, :]), (-1, -2))
                log_q = log_q + temp / K_ahead

        return x_samples, mu, var, log_q

    def compute_log_prior(self, ref_vae, x_samples, u=None):
        """
        function to compute multi-step prediction (1/K) * sum_{i=1:K} p(x_{t+k} | x_t)
        """
        dx = x_samples.shape[-1]

        if u is None:
            x_prev = x_samples[:, :-self.k_step]
            u_k = None
        else:
            x_prev = torch.cat((x_samples[:, :-self.k_step], u[:, :-self.k_step]), axis=-1)
            u_k = vectorize_x(u[..., 1:, :], self.k_step)

        if self.k_step == 1:
            log_k_step_prior = ref_vae.prior(x_prev, x_samples[..., 1:, :])

        else:
            # get vectorized k-step windows of x_samples BW x k_step x dx
            x_k = vectorize_x(x_samples[..., 1:, :], self.k_step)

            _, mu_k_ahead, var_k_ahead = ref_vae.prior.sample_k_step_ahead(x_prev.reshape(-1, 1, dx + self.du),
                                                                           self.k_step, u=u_k,
                                                                           keep_trajectory=True)
            # take mean along k-steps
            log_k_step_prior = torch.mean(Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_k), -2)

            # sum along time and dx
            log_k_step_prior = torch.sum(log_k_step_prior.reshape(x_samples.shape[-3], -1, dx), (-1, -2))

            # compute prediction for the last k-1 steps
            for i in torch.arange(self.k_step - 1):
                K_ahead = self.k_step - i - 1
                x_prev = x_samples[..., -(K_ahead + 1)]

                if u is not None:
                    x_prev = torch.cat((x_prev, u[..., -(-K_ahead + 1), :]), axis=-1)
                    u_k = u[..., -K_ahead:, :]

                _, mu_k, var_k = ref_vae.prior.sample_k_step_ahead(x_prev.unsqueeze(1), K_ahead, u=u_k,
                                                                   keep_trajectory=True)
                log_prior = torch.sum(
                    Normal(mu_k, torch.sqrt(var_k)).log_prob(x_samples[:, -K_ahead:].reshape(-1, K_ahead, dx)),
                    (-1, -2))
                log_k_step_prior = log_k_step_prior + log_prior / K_ahead

        return log_k_step_prior

    def forward(self, ref_vae, y, vel=None, u=None, beta=1.0):

        # apply transformation to data
        y_in = y if u is None else torch.cat((y, u), -1)

        y_tfm = self.f_enc(y_in)

        # pass to encoder and get samples
        x_samples, _, _, log_q = self.compute_log_q(ref_vae, y_tfm)

        log_k_step_prior = self.compute_log_prior(ref_vae, x_samples, u)

        # get likelihood in original space
        log_like = self.f_dec(x_samples, y)

        elbo = torch.mean(log_like + beta * (log_k_step_prior - log_q))

        if vel is None:
            return -elbo
        else:
            log_like_behav = self.readout_behav(x_samples, vel)
            return -elbo - torch.mean(log_like_behav)







