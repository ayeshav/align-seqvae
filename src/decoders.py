import torch
import torch.nn as nn
from torch.distributions import Normal, Binomial

Softplus = torch.nn.Softplus()
eps = 1e-6


class NormalDecoder(nn.Module):
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
        log_prob = torch.sum(Binomial(total_count=self.total_count, probs=probs).log_prob(x))
        return log_prob


class DualNormalDecoder(nn.Module):
    def __init__(self, dx, dy, dy_other, device='cpu'):
        super().__init__()
        self.device = device

        self.decoder = nn.Linear(dx, dy).to(device)
        self.f_decoder_mean = nn.Linear(dy, dy_other).to(device)

        self.logvar = nn.Parameter(0.01 * torch.randn(1, dy, device=device), requires_grad=True)
        self.f_decoder_logvar = nn.Linear(dy, dy_other).to(device)

    def compute_param(self, x, other_flag=False):
        mu = self.decoder(x)
        var = Softplus(self.logvar) + eps
        if other_flag:
            mu = self.f_decoder_mean(mu)
            var = Softplus(self.f_decoder_logvar(self.logvar)) + eps
        return mu, var

    def forward(self, samples, x, other_flag=False):
        # given samples, compute parameters of likelihood
        mu, var = self.compute_param(samples, other_flag=other_flag)

        # now compute log prob
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-1, -2))
        return log_prob


class DualBinomialDecoder(nn.Module):
    def __init__(self, dx, dy, dy_other, device='cpu', total_count=4):
        super(DualBinomialDecoder, self).__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)
        self.f_decoder = nn.Linear(dy, dy_other).to(device)
        self.total_count = total_count

    def compute_param(self, x, other_flag=False):
        log_probs = self.decoder(x)
        if other_flag:
            log_probs = self.f_decoder(log_probs)
        probs = torch.sigmoid(torch.clip(log_probs, -15, 15))
        return probs

    def forward(self, samples, x, other_flag=False):
        probs = self.compute_param(samples, other_flag=other_flag)
        log_prob = torch.sum(Binomial(total_count=self.total_count, probs=probs).log_prob(x))
        return log_prob
