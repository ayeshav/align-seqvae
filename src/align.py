import torch
import torch.nn as nn
from utils import *


class Align(nn.Module):
    def __init__(self, dy, dy_ref, decoder, linear_flag=True, k_step=1,device='cpu'):
        super().__init__()

        if linear_flag:
            self.g = nn.Linear(dy, dy_ref).to(device)
        else:
            self.g = Mlp(dy, dy_ref, 64).to(device)

        self.decoder = decoder
        self.k_step = k_step

    def compute_kl(self, ref_vae, y):

        y_tfm = self.g(y)

        x_samples, mu, var, log_q = ref_vae.compute_k_step_log_q(y_tfm)

        log_k_step_prior = ref_vae.compute_k_step_log_prior(x_samples)

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



