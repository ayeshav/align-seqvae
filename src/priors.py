import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()
eps = 1e-6


class PriorMlp(nn.Module):
    def __init__(self, dx, residual=True, fixed_variance=True, du=0, device='cpu'):
        super().__init__()

        self.dx = dx
        self.du = du
        self.residual = residual
        self.fixed_variance = fixed_variance

        if fixed_variance:
            self.logvar = nn.Parameter(-2 * torch.randn(1, dx, device=device), requires_grad=True)
            d_out = dx
        else:
            d_out = 2 * dx

        self.prior = nn.Sequential(nn.Linear(dx + du, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, d_out)).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        assert x.shape[-1] == self.dx + self.du
        out = self.prior(x)

        if self.fixed_variance:
            mu = out
            var = Softplus(self.logvar) + eps
        else:
            mu, logvar = torch.split(out, [self.dx, self.dx], -1)
            var = Softplus(logvar) + eps

        if self.residual:
            x, u = torch.split(x, [self.dx, self.du], -1)
            mu = mu + x
        return mu, var

    def forward(self, x_prev, x):
        """
        Given data, we compute the log-density of the time series
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x_prev)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-2, -1))
        return log_prob


class PriorGru(nn.Module):
    def __init__(self, dx, dh, device='cpu'):
        super().__init__()

        self.dx = dx
        self.dh = dh

        self.readin = nn.Linear(dx, dh, device=device)
        self.prior = nn.GRUCell(0, dh, device=device)
        self.readout = nn.Linear(dh, dx, device=device)

        self.logvar = nn.Parameter(-2 * torch.randn(1, dx, device=device), requires_grad=True)

        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        B, T, dx = x.shape

        h_in = self.readin(x).reshape(-1, self.dh)
        input = torch.empty(B * T, 0)

        h_out = self.prior(input, h_in)

        mu = self.readout(h_out).reshape(B, T, dx)

        var = Softplus(self.logvar) + eps

        return mu, var

    def forward(self, x_prev, x):
        """
        Given data, we compute the log-density of the time series
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x_prev)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-2, -1))
        return log_prob
