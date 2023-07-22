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
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                K_ahead = min(self.k_step, x_samples[:, t + 1:].shape[1])
                _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                log_prior = log_prior + torch.sum(
                    Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def forward(self, y, beta=1.):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return -elbo


class DualAnimalSeqVae(nn.Module):
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
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                K_ahead = min(self.k_step, x_samples[:, t + 1:].shape[1])
                _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                log_prior = log_prior + torch.sum(
                    Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def forward(self, y_ref, y_other, beta=1., align_mode=False):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density for both ref and other animal
        x_samples_ref, _, _, log_q_ref, x_samples_other, _, _, log_q_other = self.encoder(y_ref, y_other)

        # given samples, compute the log prior for reference animal
        log_prior_ref = self._prior(x_samples_ref)

        # given samples, compute the log prior for other animal
        log_prior_other = self._prior(x_samples_other)

        # given samples, compute the log likelihood for reference animal
        log_like_ref = self.decoder(x_samples_ref, y_ref)

        # given samples, compute the log likelihood for other animal
        log_like_other = self.decoder(x_samples_other, y_other, other_flag=True)

        # compute the elbo
        elbo = torch.mean(log_like_ref + log_like_other + beta * (log_prior_ref + log_prior_other - log_q_ref - log_q_other))
        return -elbo


class AlignSeqVae(nn.Module):
    def __init__(self, prior, encoder, decoder, align, device='cpu', k_step_vae=1, k_step_align=1):
        super().__init__()
        self.k_step_vae = k_step_vae
        self.k_step_align = k_step_align

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.align = align
        self.device = device

    def _prior(self, x_samples, k):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        if k == 1:
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                K_ahead = min(k, x_samples[:, t + 1:].shape[1])
                _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                log_prior = log_prior + torch.sum(
                    Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def forward(self, y_ref, y_other, beta=1., align_mode=False):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density for both ref and other animal
        x_samples_ref, _, _, log_q_ref = self.encoder(y_ref)
        x_samples_other, _, _, log_q_other = self.encoder(self.align.f_enc(y_other))

        # given samples, compute the log prior for reference animal
        log_prior_ref = self._prior(x_samples_ref, self.k_step_vae)

        # given samples, compute the log prior for other animal
        log_prior_other = self._prior(x_samples_other, self.k_step_vae)

        # given samples, compute the log likelihood for reference animal
        log_like_ref = self.decoder(x_samples_ref, y_ref)

        # given samples, compute the log likelihood for other animal
        log_like_other = self.align.compute_likelihood(self.decoder, x_samples_other, y_other)

        # compute the elbo for training the vae
        elbo = torch.mean(log_like_ref + log_like_other + beta * (log_prior_ref + log_prior_other - log_q_ref - log_q_other))

        if not align_mode:
            return -elbo
        else:
            # let's get the align loss
            log_prior_other_k_step = self._prior(x_samples_other, self.k_step_align)
            loss = torch.mean(log_like_other + log_prior_other_k_step)

            return -loss
