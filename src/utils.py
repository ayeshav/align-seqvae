import numpy.random as npr

import torch
import torch.nn as nn
from einops import rearrange

npr.seed(42)
eps = 1e-6


class SeqDataLoader:
    def __init__(self, data_tuple, batch_size, shuffle=True):
        """
        Constructor for fast data loader
        :param data_tuple: a tuple of matrices of size Batch size x Time x dy
        :param batch_size: batch size
        """
        self.shuffle = shuffle
        self.data_tuple = data_tuple
        self.batch_size = batch_size
        self.dataset_len = self.data_tuple[0].shape[0]

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
        else:
            r = torch.arange(self.dataset_len)

        self.indices = [r[j * self.batch_size: (j * self.batch_size) + self.batch_size] for j in range(self.n_batches)]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration
        idx = self.indices[self.i]
        batch = tuple([self.data_tuple[i][idx] for i in range(len(self.data_tuple))])
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


class Mlp(nn.Module):
    def __init__(self, dx, dy, dh, device='cpu'):
        super().__init__()

        self.net = nn.Sequential(*[nn.Linear(dx, dh), nn.Softplus(),
                                   nn.Linear(dh, dh), nn.Softplus(),
                                   nn.Linear(dh, dy)]).to(device)

    def forward(self, x):
        return self.net(x)


def to_device(batch, device='cpu'):

    if len(batch) > 1:
        return [batch[i].to(device) for i in range(len(batch))]
    else:
        return batch[0].to(device)


def normalize(y):
    dy = y.shape[-1]

    mu = torch.mean(y.reshape(-1, dy), 0, keepdim=True)
    sigma = torch.std(y.reshape(-1, dy), 0, keepdim=True)
    y_norm = (y - mu) / (sigma + eps)

    return y_norm


def vectorize(x, k):
    """
    function to unfold and vectorize tensor for k step prediction
    :param x: input of shape (B by T by dx)
    :param k: number of prediction steps
    :returns: tensor of shape (B x T-k) by k by dx
    """
    # get windows of size k for each trial and time point
    x_unfold = x.unfold(1, k, 1)
    return rearrange(x_unfold, 'batch time dx k -> (batch time) k dx')


def tensorize(x, b):
    """
    function to tensorize an unfolded vector f
    :param x: input of shape (B x T-k) by k by dx
    :param k: number of prediction steps
    :param b: batch size
    :returns: tensor of shape B by T by dx
    """
    return rearrange(x, '(batch time) 1 dx -> batch time dx', batch=b)
