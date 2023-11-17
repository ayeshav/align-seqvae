import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_vector_field(dynamics, xmin, xmax, ymin, ymax, device='cpu'):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    :return: a N by dy matrix
    """
    xt, yt = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    queries = np.stack([xt.ravel(), yt.ravel()]).T
    queries = torch.from_numpy(queries).float().to(device)
    with torch.no_grad():
        N = queries.shape[0]
        vel = torch.zeros(queries.shape)
        for n in tqdm(range(N)):
            vel[n, :] = (dynamics(queries[[n]]) - queries[[n]]).to('cpu')

    vel_x = vel[:, 0].reshape(xt.shape[0], xt.shape[1])
    vel_y = vel[:, 1].reshape(yt.shape[0], yt.shape[1])
    speed = torch.sqrt(vel_x ** 2 + vel_y ** 2)
    return xt, yt, vel_x, vel_y, speed


def plot_vector_field(xt, yt, vel_x, vel_y, speed, fig=None, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.streamplot(xt, yt, vel_x.numpy(), vel_y.numpy(), color=speed.numpy(), cmap='coolwarm')
    return ax


def compute_r2(y_pred, y_true):
    res = (y_pred - y_true) ** 2
    var = (y_true - y_true.mean(0, keepdim=True)) ** 2

    return 1 - res.sum(0) / var.sum(0)


def get_y_tfm(y, obj, method='align'):
    with torch.no_grad():
        if method == 'align' or method == 'align_k1':
            y_tfm = obj.g(y)

        elif method == 'mla':
            y_tfm = obj.low_d_readin_t(y)

        elif method == 'nomad':
            y_in = obj.read_in(y)
            y_tfm = obj.align(y_in)

        elif method == 'cycle-gan':
            y_tfm = obj(y)

        elif method == 'cca' or method == 'op':
            y_tfm = obj.tfm(y)

        elif method == 'retrain':
            y_tfm = y

        else:
            print('not implemented! or key error! go figure..')

    return y_tfm


def compute_prediction_r2(logrates, y_preds, T, k):
    """
    y_preds is of shape K x k_step x dy x W
    """

    dy = logrates.shape[-1]

    y_true = logrates[:, T:].unfold(1, k, 1).permute(1, 0, 3, 2)

    y_bar = logrates[:, :T].mean(1, keepdim=True)

    var = (y_true - y_bar) ** 2
    res = (y_preds - y_true) ** 2

    res = res.reshape(-1, k, dy)
    var = var.reshape(-1, k, dy)

    return 1 - res.sum(0) / var.sum(0)


def generate_k_step_pred(vae, y_test, T_train, k_step, decoder, align=None, method='align'):
    """
    function to compute k-step prediction performance
    :param vae: trained vae
    :param y_test: test observations of shape K x T x dy
    :param T_train: number of time steps for input to the encoder
    :param k_step: number of prediction steps
    :param align: optional align obj if y needs to be transformed
    :param method: possible args are 'align_k1', 'align', 'mla', 'nomad', 'cycle-gan', 'op'
    :return y_pred
    """
    y_preds = []

    with torch.no_grad():

        if method != 'retrain':
            y_test = get_y_tfm(y_test, align, method=method)

        y_in = y_test[:, :-k_step, :].unfold(1, T_train, 1).permute(0, 3, 2, 1)

        for i in range(y_in.shape[-1]):

            x = vae.encoder.sample(y_in[..., i])[0]
            x_k_ahead, _, _ = vae.prior.sample_k_step_ahead(x[:, -1, :][:, np.newaxis, :], k_step,
                                                            keep_trajectory=True)
            y_pred = decoder.compute_param(x_k_ahead)  # shape K x k_step x dy

            if type(y_pred) == tuple:
                y_pred = y_pred[0]

            y_preds.append(y_pred)
    return torch.stack(y_preds)
