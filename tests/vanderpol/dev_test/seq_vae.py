import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()
eps = 1e-6


class SeqVae(nn.Module):
    def __init__(self, prior, encoder, decoder, device='cpu', k_step=1):
        super().__init__()
        self.k_step = k_step

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _prior(self, x_samples):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        if self.k_step == 1:
            if len(x_samples.shape) > 3:
                log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                             torch.ones(1, device=self.device)).log_prob(x_samples[:, :, 0]), -1)
                log_prior = log_prior + self.prior(x_samples[:, :, :-1], x_samples[:, :, 1:])
            else:
                log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                             torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
                log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                if len(x_samples.shape) > 3:
                    K_ahead = min(self.k_step, x_samples[:, :, t + 1:].shape[1])
                    _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, :, t], K_ahead)
                    log_prior = log_prior + torch.sum(
                        Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, :, t + K_ahead]), -1)
                else:
                    K_ahead = min(self.k_step, x_samples[:, t + 1:].shape[1])
                    _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                    log_prior = log_prior + torch.sum(
                        Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def forward(self, y, inp_tfm=None, beta=1., n_samples=1):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        if inp_tfm is None:
            y_enc = y
        else:
            y_enc = inp_tfm(y)
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y_enc, n_samples=n_samples)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return -elbo
