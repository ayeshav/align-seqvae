import math
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from seq_vae import SeqVae
from torch.utils.data import Dataset
from tqdm import tqdm


class SeqDataLoader:
    def __init__(self, data_tuple, batch_size, shuffle=False):
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
    opt = torch.optim.AdamW(params=vae.parameters(), lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for y in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device))
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def compute_map_mse(ref_vae, prior, linear_map, y):
    """
    loss for alignment between datasets
    ref_vae: pre-trained vae
    prior: pre-trained prior
    linear_map: linear alignment matrix of size dy x dy_ref
    y: new dataset to be aligned of shape K x T x dy TODO: shouldn't it be Time by K by dy?
    """
    assert isinstance(prior, Prior)
    assert isinstance(ref_vae, SeqVae)

    dy, dy_ref = linear_map.shape  # Assumption right now is that dy and dy_ref are of same dimension and no translations
    y_tfm = y @ linear_map  # apply linear transformation to new dataset

    encoder_params, likelihood_params, log_prior = ref_vae(y_tfm, prior)  # for the given dataset

    x_samples = encoder_params[2]  # sample from the encoder after doing the transformation

    # measure samples under the log prior to make sure it matches up with the learned generative model
    log_prior = ref_vae._prior(x_samples, prior)

    # now, we want to make sure we can reconstruct the original data, NOT y_tfm
    # TODO: can we show that reconstructing y_tfm is equivalent to learning to reconstruct y???
    mu_like_tfm, sigma_like_tfm, _ = likelihood_params

    # let's commit a sin and work with inverses
    inv_linear_map = torch.linalg.pinv(linear_map)
    mu_like = mu_like_tfm @ inv_linear_map
    sigma_like = inv_linear_map.T @ (sigma_like_tfm * torch.eye(dy_ref)) @ inv_linear_map
    sigma_like = sigma_like + 1e-5 * torch.eye(dy)  # for numerical stability
    log_like = torch.sum(MultivariateNormal(mu_like, covariance_matrix=sigma_like).log_prob(y))

    loss = torch.mean(log_like + log_prior)
    return -loss


def train_invertible_mapping(epochs, ref_vae, prior, y, dy_ref):
    """
    training function for learning linear alignment and updating prior params
    """
    dy = y.shape[2]
    linear_map = nn.Parameter(torch.randn(dy, dy_ref) / math.sqrt(dy), requires_grad=True)

    param_list = [linear_map]
    training_losses = []
    opt = torch.optim.Adam(params=param_list, lr=1e-3)

    for _ in tqdm(range(epochs)):

        # for y_b,  in data:
        opt.zero_grad()
        loss = compute_map_mse(ref_vae, prior, linear_map, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            training_losses.append(loss.item())

    return linear_map, training_losses


def obs_alignment(ref_res, prior, y, y_ref, epochs=20, update_prior=True):
    """
    ref_res: reference vae trained on y_ref
    prior: trained prior on y_ref
    y: new data to be aligned of shape K x T x dy
    y_ref: reference dataset of shape K x T x dy_ref

    returns linear map, rp_mat and trained prior
    """
    T, N, dy = y.shape
    dy_ref = y_ref.shape[2]

    if dy != dy_ref:
        rp_mat = torch.randn(dy, dy_ref) * (1 / dy_ref)

    linear_map, prior = train_invertible_mapping(epochs, ref_res, prior, y, y_ref, rp_mat, update_prior)
    return linear_map, rp_mat, prior

