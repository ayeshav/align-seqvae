import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


class Encoder(nn.Module):
    def __init__(self, dx, w, h):
        super().__init__()
        self.dx = dx

        self.encoder = nn.Sequential(nn.Linear(w * h, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, dx * 2))

    def forward(self, x):
        output = self.encoder(x)

        return output


class Decoder(nn.Module):
    def __init__(self, dx, w, h):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(dx, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, w * h))

    def forward(self, x):
        return self.decoder(x)


class VAE(nn.Module):
    def __init__(self, dx, w, h):
        super().__init__()
        self.dx = dx

        self.encoder = Encoder(dx, w, h)
        self.decoder = Decoder(dx, w, h)

    def forward(self, y):

        encoder_output = self.encoder(y)
        mu, var = torch.split(encoder_output, [self.dx, self.dx], -1)

        "generate samples from encoder output"
        x_samples = mu + torch.randn(mu.shape) * torch.sqrt(torch.exp(var))

        decoder_prob = torch.sigmoid(self.decoder(x_samples))

        return mu, torch.exp(var), x_samples, decoder_prob
