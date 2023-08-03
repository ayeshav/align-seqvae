import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import compute_wasserstein

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
    def __init__(self, prior, encoder, decoder, align,
                 device='cpu', k_step=1):
        super().__init__()
        self.k_step = k_step

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.align = align
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

    def ref_animal_loss(self, y_ref, beta=1.):
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y_ref)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y_ref)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return x_samples, -elbo

    def other_animal_loss(self, y_other, beta=1.):
        # first pass data from f_enc
        y_hat = self.align.f_enc(y_other)

        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y_hat)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # can we reconstruct y_hat using regular decoder

        # log_like = self.decoder(x_samples, y_hat)
        #
        # # compute the likelihood using animal specific likelihood
        # log_like = log_like + self.other_animal_decoder(x_samples, y_other)

        # for now use the f_dec directly
        log_like = self.align.f_dec(x_samples, y_other)

        # compute elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return x_samples, -elbo

    def forward(self, y_ref, y_other, beta=1., align_mode=False, reg_weight=100):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # ref animal elbo
        x_ref_samples, ref_neg_elbo = self.ref_animal_loss(y_ref, beta=beta)

        mu_ref = torch.mean(x_ref_samples.view(-1, x_ref_samples.shape[-1]), 0)
        cov_ref = torch.cov(x_ref_samples.view(-1, x_ref_samples.shape[-1]).T)

        # other animal elbo
        x_other_samples, other_neg_elbo = self.other_animal_loss(y_other, beta=beta)
        mu_other = torch.mean(x_other_samples.view(-1, x_other_samples.shape[-1]), 0)
        cov_other = torch.cov(x_other_samples.view(-1, x_other_samples.shape[-1]).T)

        # compute Wassertein
        reg = compute_wasserstein(mu_ref, cov_ref,
                                  mu_other, cov_other)

        if not align_mode:
            return ref_neg_elbo + other_neg_elbo + reg_weight * reg
        else:
            return other_neg_elbo