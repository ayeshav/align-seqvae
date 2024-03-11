import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import vectorize
from vae_utils import *

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

    def compute_k_step_log_q(self, y):

        x_samples, mu, var = self.encoder.sample(y)

        if self.k_step == 1:
            log_q = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_samples), (-2, -1))
        else:
            log_q = compute_k_step_log_q(x_samples, mu, var, self.k_step, method='multi_step')

        return x_samples, mu, var, log_q

    def compute_k_step_log_prior(self, x_samples):
        """
        function to compute multi-step prediction (1/K) * sum_{i=1:K} p(x_{t+k} | x_t)
        """
        if self.k_step == 1:
            log_k_step_prior = self.prior(x_samples[:, :-1], x_samples[:, 1:, :self.prior.dx])
        else:
            log_k_step_prior = compute_k_step_prior(x_samples, self.k_step, self.prior, method='multi_step')

        return log_k_step_prior

    def forward(self, y, beta=1.):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, _, _, log_q = self.compute_k_step_log_q(y)

        # given samples, compute the log prior
        log_prior = self.compute_k_step_log_prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return -elbo


class CondSeqVae(SeqVae):
    def __init__(self, prior, encoder, decoder, readout_behav=None, device='cpu', k_step=1):
        super().__init__(prior=prior, encoder=encoder, decoder=decoder, device=device)
        self.k_step = k_step

        self.readout_behav = readout_behav
        self.device = device

    def forward(self, batch, beta=1.):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :param u: U is a tensor of input of size B x T x Du
        :param y_behav: tensor of behavior (example, hand velocity) of size B x T x Y_behav
        :return:
        """
        if len(batch) > 2:
            y, u, y_behav = batch
        else:
            y, u = batch

        # pass data through encoder and get mean, variance, samples and log density
        x_samples, _, _, log_q = self.compute_k_step_log_q(torch.cat((y, u), -1))

        # given samples, compute the log prior
        log_prior = self.compute_k_step_log_prior(torch.cat((x_samples, u), -1))

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))

        if len(batch) > 2:
            log_like_behav = self.readout_behav(x_samples, y_behav)

            return -elbo - torch.mean(log_like_behav)
        else:
            return -elbo

