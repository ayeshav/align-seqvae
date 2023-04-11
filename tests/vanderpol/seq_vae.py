import torch
import torch.nn as nn


Softplus = torch.nn.Softplus()


class Prior(nn.Module):
    def __init__(self, dx):
        super().__init__()

        self.dx = self.dx
        self.prior = nn.Sequential(nn.Linear(dx, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 2*dx))

    def forward(self, x):
        out = self.prior(x)
        mu, logvar = torch.split(out, [self.x, self.x], -1)
        var = Softplus(logvar) + 1e-6

        return mu, var


class Encoder(nn.Module):
    def __init__(self, dy, dx, dh):
        super().__init__()

        self.dh = dh
        self.dx = dx

        self.gru = nn.GRU(input_size=dy, hidden_size=dh, bidirectional=True)
        self.readout = nn.Linear(dh, dx)

    def forward(self, x):
        h, h_l = self.gru(x)
        h = h.view(x.shape[0], x.shape[1], 2, self.dh)[:, :, 1, :]
        out = self.readout(h)

        dim = int(self.dx / 2)
        mu, logvar = torch.split(out, [dim, dim], -1)
        var = Softplus(logvar) + 1e-6
        return mu, var


class Decoder(nn.Module):
    def __init__(self, dx, dy):
        super().__init__()

        self.decoder = nn.Sequential(nn.Linear(dx, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, dy))

    def forward(self, x):
        return self.decoder(x)


class SeqVae(nn.Module):
    def __init__(self, dx, dy, dh_e):
        super().__init__()

        self.dx = dx

        self.encoder = Encoder(dy, dx, dh_e)
        self.decoder = Decoder(dx, dy)

    def forward(self, y):

        encoder_output = self.encoder(y)
        mu, var = torch.split(encoder_output, [self.dx, self.dx], -1)
        # torch.clamp(var, 1e-3)

        logvar = Softplus(var) + 1e-6

        "generate samples from encoder output"
        x_samples = mu + torch.randn(mu.shape) * torch.sqrt(logvar)

        decoder_out = self.decoder(x_samples)

        return mu, logvar, x_samples, decoder_out







