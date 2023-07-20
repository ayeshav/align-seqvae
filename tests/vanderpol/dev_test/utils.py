import torch
import torch.nn as nn
from seq_vae import SeqVae
from tqdm import tqdm


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


def compute_wasserstein(mu_s, cov_s, mu_t, cov_t):
    "function to compute wasserstein assuming P_s and P_t are independent gaussians"

    matrix_sqrt = torch.sqrt(torch.diag(cov_t))
    ind_cov_s = torch.diag((torch.diag(cov_s)))

    w2 = torch.sum((mu_s - mu_t) ** 2) + torch.trace(cov_s) + torch.trace(cov_t) - \
         2 * torch.trace(torch.sqrt(torch.diag(matrix_sqrt) @ ind_cov_s @ torch.diag(matrix_sqrt)))

    return w2
