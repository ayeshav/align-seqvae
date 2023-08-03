import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()
eps = 1e-6


class Prior(nn.Module):
    def __init__(self, dx, residual=True, fixed_variance=True, device='cpu'):
        super().__init__()

        self.dx = dx
        self.residual = residual
        self.fixed_variance = fixed_variance

        if fixed_variance:
            self.logvar = nn.Parameter(0.01 * torch.randn(1, dx, device=device), requires_grad=True)
            d_out = dx
        else:
            d_out = 2 * dx

        self.prior = nn.Sequential(nn.Linear(dx, 256),
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
        assert x.shape[-1] == self.dx
        out = self.prior(x)

        if self.fixed_variance:
            mu = out
            var = Softplus(self.logvar) + eps
        else:
            mu, logvar = torch.split(out, [self.dx, self.dx], -1)
            var = Softplus(logvar) + eps

        if self.residual:
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

    def sample_k_step_ahead(self, x, K, keep_trajectory=False):
        x_trajectory = []
        means = []
        vars = []
        for t in range(K):
            if t == 0:
                mu, var = self.compute_param(x)
            else:
                mu, var = self.compute_param(x_trajectory[-1])
            means.append(mu)
            vars.append(var)

            x_trajectory.append(mu + torch.sqrt(var) * torch.randn(x.shape, device=x.device))
        if keep_trajectory:
            return x_trajectory, means, vars
        else:
            return x_trajectory[-1], means[-1], vars[-1]


class PriorVanderPol(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.dt = 0.1 * torch.ones(1, device=device)
        self.var = (self.dt * 0.5) ** 2 * torch.ones(1, device=device)
        self.mu = 1.5 * torch.ones(1, device=device)

    def compute_velocity(self, x):
        if len(x.shape) == 2:
            vel_x = self.mu * (x[:, 0] - x[:, 0] ** 3 / 3 - x[:, 1])
            vel_y = x[:, 0] / self.mu
        elif len(x.shape) == 3:
            vel_x = self.mu * (x[:, :, 0] - x[:, :, 0] ** 3 / 3 - x[:, :, 1])
            vel_y = x[:, :, 0] / self.mu
        else:
            vel_x = self.mu * (x[:, :, :, 0] - x[:, :, :, 0] ** 3 / 3 - x[:, :, :, 1])
            vel_y = x[:, :, :, 0] / self.mu
        return self.dt * torch.stack((vel_x, vel_y), -1)

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu = x + self.compute_velocity(x)
        var = self.var
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

    def sample_k_step_ahead(self, x, K, keep_trajectory=False):
        x_trajectory = []
        means = []
        vars = []
        for t in range(K):
            if t == 0:
                mu, var = self.compute_param(x)
            else:
                mu, var = self.compute_param(x_trajectory[-1])
            means.append(mu)
            vars.append(var)

            x_trajectory.append(mu + torch.sqrt(var) * torch.randn(x.shape, device=x.device))
        if keep_trajectory:
            return x_trajectory, means, vars
        else:
            return x_trajectory[-1], means[-1], vars[-1]