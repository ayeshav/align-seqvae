import torch
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


def vae_training(vae, train_dataloader, n_epochs=100, lr=5e-4, weight_decay=1e-4):
    """
    function that will train a vae
    :param vae: a SeqVae object
    :param train_dataloader: a dataloader object
    :param n_epochs: Number of epochs to train for
    :param lr: learning rate of optimizer
    :param weight_decay: value of weight decay
    :return: trained vae and training losses
    """
    assert isinstance(vae, SeqVae)
    assert train_dataloader.shuffle
    assert isinstance(train_dataloader, SeqDataLoader)
    opt = torch.optim.AdamW(params=vae.parameters(), lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device))
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def compute_wasserstein(mu_s, cov_s, mu_t, cov_t):
    "function to compute wasserstein assuming P_s and P_t are independent gaussians"

    matrix_sqrt = torch.sqrt(torch.diag(cov_t))
    ind_cov_s = torch.diag((torch.diag(cov_s)))

    w2 = torch.sum((mu_s - mu_t) ** 2) + torch.trace(cov_s) + torch.trace(cov_t) - \
         2 * torch.trace(torch.sqrt(torch.diag(matrix_sqrt) @ ind_cov_s @ torch.diag(matrix_sqrt)))

    return w2


# def get_matrix_sqrt(cov):
#
#     lam, Q = torch.linalg.eigh(cov)
#     return Q@torch.diag(torch.sqrt(lam))@Q.T

