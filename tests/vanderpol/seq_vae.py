import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, Bernoulli

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
                                   nn.Softplus(),
                                   nn.Linear(256, 256),
                                   nn.Softplus(),
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
        # if len(x.shape) > 3:
        #     mu, var = self.compute_param(x[:, :, :-1])
        #     log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x[:, :, 1:]), (-2, -1))
        # else:
        #     mu, var = self.compute_param(x[:, :-1])
        #     log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x[:, 1:]), (-2, -1))
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


class Encoder(nn.Module):
    def __init__(self, dy, dx, dh, prior_func=None, device='cpu'):
        super().__init__()

        self.dh = dh
        self.dx = dx
        self.prior_func = prior_func

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

        if self.prior_func is not None:
            # pass mean through dynamics to get its parameters
            mu_prior, var_prior = self.prior_func(mu)

            # combine
            tau = 1 / (1 / var + 1 / var_prior)
            mu_updated = tau * (mu / var + mu_prior / var_prior)
            mu = mu_updated
            var = tau
        return mu, var

    def sample(self, x, n_samples=1):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=self.device)
        return samples.squeeze(0), mu, var

    def forward(self, x, n_samples=1):
        """
        Given a batch of time series, sample and compute the log density
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        # compute parameters and sample
        samples, mu, var = self.sample(x, n_samples=n_samples)
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


class PoissonDecoder(nn.Module):
    def __init__(self, dx, dy, device='cpu'):
        super().__init__()
        self.device = device
        self.decoder = nn.Sequential(nn.Linear(dx, dy)).to(device)

    def compute_param(self, x):
        log_rates = self.decoder(x)
        rates = torch.nn.functional.softplus(log_rates)
        return rates

    def forward(self, samples, x):
        rates = self.compute_rate(samples)
        log_prob = torch.sum(Poisson(rates).log_prob(x), (-1, -2))
        return log_prob


class BernoulliDecoder(nn.Module):
    def __init__(self, dx, dy, device='cpu'):
        super(BernoulliDecoder, self).__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)

    def compute_param(self, x):
        log_probs = self.decoder(x)
        probs = torch.sigmoid(torch.clip(log_probs, -15, 15))
        return probs

    def forward(self, samples, x):
        probs = self.compute_param(samples)
        log_prob = torch.sum(Bernoulli(probs=probs).log_prob(x))
        return log_prob


class SeqVae(nn.Module):
    def __init__(self, dx, dy, dh_e, dy_out=None, likelihood='Bernoulli', fixed_variance=True, fancy=True, device='cpu'):
        super().__init__()

        self.dx = dx

        self.prior = Prior(dx, fixed_variance=fixed_variance, device=device)
        self.encoder = Encoder(dy, dx, dh_e, device=device,
                               prior_func=self.prior.compute_param if fancy else None)
        self.device = device

        if dy_out is None:
            dy_out=dy

        if likelihood == 'Normal':
            self.decoder = Decoder(dx, dy_out, device=device)
        elif likelihood == 'Poisson':
            self.decoder = PoissonDecoder(dx, dy_out, device=device)
        elif likelihood == 'Bernoulli':
            self.decoder = BernoulliDecoder(dx, dy_out, device=device)

    def _prior(self, x_samples):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        if len(x_samples.shape) > 3:
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, :, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :, :-1], x_samples[:, :, 1:])
        else:
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        return log_prior

    # def compute_k_step_pred(self, x, k):
    #     # TODO: we will use this for training the SeqVae when using multiple animals so keep like this and will edit later
    #     x_unfold = torch.Tensor.unfold(x, 1, k+1, 1)
    #     mu, var = self.prior.compute_param(x_unfold[..., 0])
    #
    #     log_prob = 0
    #
    #     for i in range(k):
    #         log_prob = log_prob + torch.mean(torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_unfold[:, ..., i + 1]), (-2, -1)))
    #
    #         x_samples_next = mu + torch.sqrt(var) * torch.randn(mu.shape, device=self.device)
    #         mu, var = self.prior.compute_param(x_samples_next)
    #
    #     return log_prob

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
