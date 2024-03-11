import torch
import torch.nn as nn
from torch.distributions import Normal, Binomial, Poisson

Softplus = torch.nn.Softplus()
eps = 1e-6


class NormalDecoder(nn.Module):
    def __init__(self, dx, dy, device='cpu'):
        super().__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)
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


class BinomialDecoder(nn.Module):
    def __init__(self, dx, dy, device='cpu', total_count=4):
        super(BinomialDecoder, self).__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)
        self.total_count = total_count

    def compute_param(self, x):
        log_probs = self.decoder(x)
        probs = torch.sigmoid(torch.clip(log_probs, -15, 15))
        return probs

    def forward(self, samples, x):
        probs = self.compute_param(samples)
        log_prob = torch.sum(Binomial(total_count=self.total_count, probs=probs).log_prob(x), (-1,-2))
        return log_prob


class PoissonDecoder(nn.Module):
    def __init__(self, dx, dy, device='cpu'):
        super(PoissonDecoder, self).__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)

    def compute_param(self, x):
        log_rates = self.decoder(x)
        rates = Softplus(log_rates) + eps
        return rates

    def forward(self, samples, x):
        rates = self.compute_param(samples)
        log_prob = torch.sum(Poisson(rates).log_prob(x), (-1,-2))
        return log_prob