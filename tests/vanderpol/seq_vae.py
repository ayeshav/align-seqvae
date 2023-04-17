import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()


class Prior(nn.Module):
    def __init__(self, dx):
        super().__init__()

        self.dx = dx
        self.prior = nn.Sequential(nn.Linear(dx, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 2*dx))

    def forward(self, x):
        out = self.prior(x)
        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + 1e-6

        return mu, var


class Encoder(nn.Module):
    def __init__(self, dy, dx, dh):
        super().__init__()

        self.dh = dh
        self.dx = dx

        self.gru = nn.GRU(input_size=dy, hidden_size=dh, bidirectional=True)
        self.readout = nn.Linear(dh, 2*dx)

    def forward(self, x):
        h, h_l = self.gru(x)
        h = h.view(x.shape[0], x.shape[1], 2, self.dh)[:, :, 1, :]
        out = self.readout(h)

        dim = int(self.dx)
        mu, logvar = torch.split(out, [dim, dim], -1)
        var = Softplus(logvar) + 1e-6
        return mu, var


class Decoder(nn.Module):
    def __init__(self, dx, dy):
        super().__init__()

        self.decoder = nn.Sequential(nn.Linear(dx, dy))

    def forward(self, x):
        return self.decoder(x)


class SeqVae(nn.Module):
    def __init__(self, dx, dy, dh_e):
        super().__init__()

        self.dx = dx

        self.encoder = Encoder(dy, dx, dh_e)
        self.decoder = Decoder(dx, dy)

        self.obs_var = nn.Parameter(torch.ones(1, dy))

    def sample(self, y):
        mu, logvar = self.encoder(y)

        "generate samples from encoder output"
        x_samples = mu + torch.randn(mu.shape) * torch.sqrt(logvar)

        log_q = torch.sum(torch.sum(Normal(mu, logvar).log_prob(x_samples), -1), 0)

        return x_samples, log_q

    def _likelihood(self, y, x_samples):
        mu_y = self.decoder(x_samples)
        var_y = Softplus(self.obs_var) + 1e-6

        ll = torch.sum(torch.sum(Normal(mu_y, var_y).log_prob(y), -1), 0)

        return ll

    def _kl(self, x_samples, log_q, prior):

        log_prior = torch.sum(Normal(0, 1).log_prob(x_samples[0]), -1)

        mu, var = prior(x_samples[:-1])
        log_prior = log_prior + torch.sum(torch.sum(Normal(mu, var).log_prob(x_samples[1:]),-1), 0)

        return log_q - log_prior

    def forward(self, y, prior):

        x_samples, log_q = self.sample(y)

        ll = self._likelihood(y, x_samples)
        kl = self._kl(x_samples, log_q, prior)

        return ll, kl






