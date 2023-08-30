import torch
import torch.nn as nn
from torch.distributions import Normal

Softplus = torch.nn.Softplus()
eps = 1e-6


class Encoder(nn.Module):
    def __init__(self, dy, dx, dh, device='cpu'):
        super().__init__()

        self.dh = dh
        self.dx = dx

        # GRU expects batch to be the first dimension
        self.gru = nn.GRU(input_size=dy, hidden_size=dh,
                          bidirectional=True, batch_first=True).to(device)
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


class EmbeddingEncoder(nn.Module):
    def __init__(self, dy, dx, dh, d_embed, device='cpu'):
        super().__init__()

        self.dh = dh
        self.dx = dx
        self.d_embed = d_embed

        # embed network
        self.embed_network = nn.Sequential(*[nn.Linear(dy, dh),
                                             nn.ReLU(),
                                             nn.Linear(dh, dh),
                                             nn.ReLU(),
                                             nn.Linear(dh, d_embed)]).to(device)

        # GRU expects batch to be the first dimension
        self.gru = nn.GRU(input_size=d_embed, hidden_size=dh,
                          bidirectional=True, batch_first=True).to(device)
        self.readout = nn.Linear(2 * dh, 2 * dx).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        # data that has to be aligned is already passed through the f_enc
        if x.shape[-1] != self.d_embed:
            x = self.embed_network(x)

        h, _ = self.gru(x)

        h = h.view(x.shape[0], x.shape[1], 2, self.dh)
        h_cat = torch.cat((h[:, :, 0], h[:, :, 1]), -1)  # TODO: can we achieve this with one view
        out = self.readout(h_cat)

        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + eps
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


class DualAnimalEncoder(nn.Module):
    def __init__(self, dy, dy_other, dx, dh, device='cpu'):
        """
        Constructor for an encoder with two animals
        :param dy: observation dimension of reference animal
        :param dy_other: observation dimension of second animal
        :param dx: dimension of latent space
        :param dh: number  of hidden units
        :param device:
        """

        super().__init__()

        self.dh = dh
        self.dx = dx
        self.dy_other = dy_other

        self.f_enc = nn.Linear(dy_other, dy).to(device)

        # GRU expects batch to be the first dimension
        self.gru = nn.GRU(input_size=dy, hidden_size=dh,
                          bidirectional=True, batch_first=True).to(device)
        self.readout = nn.Linear(2 * dh, 2 * dx).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """

        if x.shape[-1] == self.dy_other:
            x = self.f_enc(x)

        h, _ = self.gru(x)

        h = h.view(x.shape[0], x.shape[1], 2, self.dh)
        h_cat = torch.cat((h[:, :, 0], h[:, :, 1]), -1)  # TODO: can we achieve this with one view
        out = self.readout(h_cat)

        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + eps
        return mu, var

    def sample(self, x, n_samples=1):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=self.device)
        return samples.squeeze(0), mu, var

    def forward(self, x_ref, x_other, n_samples=1):
        """
        Given a batch of time series, sample and compute the log density
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        # compute parameters and sample for reference animal
        samples_ref, mu_ref, var_ref = self.sample(x_ref, n_samples=n_samples)
        log_prob_ref = torch.sum(Normal(mu_ref, torch.sqrt(var_ref)).log_prob(samples_ref), (-2, -1))

        # compute parameters and sample for other animal
        samples_other, mu_other, var_other = self.sample(x_other, n_samples=n_samples)
        log_prob_other = torch.sum(Normal(mu_other, torch.sqrt(var_other)).log_prob(samples_other), (-2, -1))
        return samples_ref, mu_ref, var_ref, log_prob_ref, samples_other, mu_other, var_other, log_prob_other


class DualAnimalEmbeddingEncoder(nn.Module):
    def __init__(self, dy, dy_other, dx, dh, d_embed, device='cpu'):
        """
        Constructor for an encoder with two animals
        :param dy: observation dimension of reference animal
        :param dy_other: observation dimension of second animal
        :param dx: dimension of latent space
        :param dh: number  of hidden units
        :param device:
        """

        super().__init__()

        self.dh = dh
        self.dx = dx
        self.dy_other = dy_other
        self.d_embed = d_embed

        # embed network
        self.embed_network = nn.Sequential(*[nn.Linear(dy, dh),
                                             nn.ReLU(),
                                             nn.Linear(dh, dh),
                                             nn.ReLU(),
                                             nn.Linear(dh, d_embed)]).to(device)

        self.f_enc = nn.Sequential(*[nn.Linear(dy_other, dh),
                                     nn.ReLU(),
                                     nn.Linear(dh, dh),
                                     nn.ReLU(),
                                     nn.Linear(dh, d_embed)]).to(device)

        # GRU expects batch to be the first dimension
        self.gru = nn.GRU(input_size=d_embed, hidden_size=dh,
                          bidirectional=True, batch_first=True).to(device)
        self.readout = nn.Linear(2 * dh, 2 * dx).to(device)
        self.device = device

    def compute_param(self, x):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """

        if x.shape[-1] == self.dy_other:
            x = self.f_enc(x)
        else:
            x = self.embed_network(x)

        h, _ = self.gru(x)

        h = h.view(x.shape[0], x.shape[1], 2, self.dh)
        h_cat = torch.cat((h[:, :, 0], h[:, :, 1]), -1)  # TODO: can we achieve this with one view
        out = self.readout(h_cat)

        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
        var = Softplus(logvar) + eps
        return mu, var

    def sample(self, x, n_samples=1):
        """
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        mu, var = self.compute_param(x)
        samples = mu + torch.sqrt(var) * torch.randn([n_samples] + list(mu.shape), device=self.device)
        return samples.squeeze(0), mu, var

    def forward(self, x_ref, x_other, n_samples=1):
        """
        Given a batch of time series, sample and compute the log density
        :param x: X is a tensor of observations of shape Batch by Time by Dimension
        :return:
        """
        # compute parameters and sample for reference animal
        samples_ref, mu_ref, var_ref = self.sample(x_ref, n_samples=n_samples)
        log_prob_ref = torch.sum(Normal(mu_ref, torch.sqrt(var_ref)).log_prob(samples_ref), (-2, -1))

        # compute parameters and sample for other animal
        samples_other, mu_other, var_other = self.sample(x_other, n_samples=n_samples)
        log_prob_other = torch.sum(Normal(mu_other, torch.sqrt(var_other)).log_prob(samples_other), (-2, -1))
        return samples_ref, mu_ref, var_ref, log_prob_ref, samples_other, mu_other, var_other, log_prob_other
