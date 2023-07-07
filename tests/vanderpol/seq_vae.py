import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()
eps = 1e-6


class Prior(nn.Module):
    def __init__(self, dx, residual=True, device='cpu'):
        super().__init__()

        self.dx = dx
        self.residual = residual
        self.prior = nn.Sequential(nn.Linear(dx, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 2 * dx)).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        assert x.shape[-1] == self.dx
        out = self.prior(x)
        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + eps

        if self.residual:
            mu = mu + x
        return mu, var

    def forward(self, x):
        """
        Given data, we compute the log-density of the time series
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x[:, :-1])
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x[:, 1:]), (-2, -1))
        return log_prob


class Encoder(nn.Module):
    def __init__(self, dy, dx, dh, device='cpu'):
        super().__init__()

        self.dh = dh
        self.dx = dx

        # GRU expects batch to be the first dimension
        self.gru = nn.GRU(input_size=dy, hidden_size=dh, bidirectional=True).to(device)
        self.readout = nn.Linear(2 * dh, 2 * dx).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        h, _ = self.gru(x)

        h = h.view(x.shape[0], x.shape[1], 2, self.dh)
        h_cat = torch.cat((h[:, :, 0], h[:, :, 1]), -1)  # TODO: can we achieve this with one view
        out = self.readout(h_cat)

        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + eps
        return mu, var

    def sample(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x)
        samples = mu + torch.sqrt(var) * torch.randn(mu.shape, device=self.device)
        return samples, mu, var

    def forward(self, x):
        """
        Given a batch of time series, sample and compute the log density
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """

        # compute parameters and sample
        samples, mu, var = self.sample(x)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(samples), (-2, -1))
        return samples, mu, var, log_prob


class Decoder(nn.Module):
    def __init__(self, dx, dy, device='cpu'):
        super().__init__()
        self.device = device
        self.decoder = nn.Sequential(nn.Linear(dx, dy)).to(device)
        self.logvar = nn.Parameter(0.01 * torch.randn(1, dy, device=device), requires_grad=True)

    def compute_param(self, x):
        mu = self.decoder(x)
        var = Softplus(self.logvar) + eps
        return mu, var

    def forward(self, samples, x):
        # given samples, compute parameters of likelihood
        mu, var = self.compute_param(samples)

        # now compute log prob
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-1, -2))
        return log_prob


class SeqVae(nn.Module):
    def __init__(self, dx, dy, dh_e, device='cpu'):
        super().__init__()

        self.dx = dx

        self.encoder = Encoder(dy, dx, dh_e, device=device)
        self.prior = Prior(dx, device=device)
        self.decoder = Decoder(dx, dy, device=device)
        self.device = device

    def _prior(self, x_samples):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                     torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
        log_prior = log_prior + self.prior(x_samples)
        return log_prior

    def forward(self, y):
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
        elbo = torch.mean(log_like + log_prior - log_q)
        return -elbo
