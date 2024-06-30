import torch
import torch.nn as nn
from torch.distributions import Normal

from src.utils import *
from src.vae_utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, decoder, linear_flag=True, k_step=1,device='cpu'):
        super().__init__()

        if linear_flag:
            self.g = nn.Linear(dy, dy_ref).to(device)
        else:
            self.g = Mlp(dy, dy_ref, 64).to(device)

        self.decoder = decoder
        self.k_step = k_step

    def compute_k_step_log_q(self, ref_vae, y, u=None):

        x_samples, mu, var = ref_vae.encoder.sample(y, u=u)

        if self.k_step == 1:
            log_q = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_samples), (-2, -1))
        else:
            log_q = compute_k_step_log_q(x_samples, mu, var, self.k_step)

        return x_samples, mu, var, log_q

    def compute_k_step_log_prior(self, ref_vae, x_samples):
        """
        function to compute multi-step prediction (1/K) * sum_{i=1:K} p(x_{t+k} | x_t)
        """
        if self.k_step == 1:
            log_k_step_prior = ref_vae.prior(x_samples[:, :-1], x_samples[:, 1:, :ref_vae.prior.dx])
        else:
            log_k_step_prior = compute_k_step_prior(x_samples, self.k_step, ref_vae.prior)

        return log_k_step_prior

    def compute_kl(self, ref_vae, y):

        y_tfm = self.g(y)

        x_samples, mu, var, log_q = self.compute_k_step_log_q(ref_vae, y_tfm)

        log_k_step_prior = self.compute_k_step_log_prior(ref_vae, x_samples)

        kl_d = log_k_step_prior - log_q

        return kl_d, x_samples

    def forward(self, ref_vae, y, beta=1.0):

        kl_d, x_samples = self.compute_kl(ref_vae, y)

        # get likelihood in original space
        log_like = self.decoder(x_samples, y)

        elbo = torch.mean(log_like + beta * kl_d)
        return -elbo


class CondAlign(Align):
    def __init__(self, dy, dy_ref, decoder, readout_behav=None, du=0,
                 linear_flag=True, k_step=1, device='cpu'):
        super().__init__(dy=dy+du, dy_ref=dy_ref, decoder=decoder,
                         linear_flag=linear_flag, k_step=k_step,
                         device=device)

        self.readout_behav = readout_behav

    def forward(self, ref_vae, batch, beta=1.0):

        if len(batch) > 2:
            y, u, y_behav = batch
        else:
            y, u = batch

        kl_d, x_samples = self.compute_kl(ref_vae, torch.cat((y, u), -1))

        log_like = self.decoder(x_samples, y)

        elbo = torch.mean(log_like + beta * kl_d)

        if len(batch) < 2:
            return -elbo
        else:
            log_like_behav = self.readout_behav(x_samples, y_behav)
            return -elbo - torch.mean(log_like_behav)



