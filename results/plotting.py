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


def compute_k_step_pred(vae, y_test, T_train, k_step=30, distribution='Normal', logrates=None, align=None):
    """
    function to compute k-step prediction performance
    :param vae: trained vae
    :param y_test: test observations of shape K x T x dy
    :param T_train: number of time steps for input to the encoder
    :param k_step: number of prediction steps
    :param distribution: likelihood distribution
    :param logrates: test logrates torch.sigmoid(x@C + b) if distribution is Binomial
    :param align: optional align if y needs to be transformed

    :return r2 of shape W x k_step x dy, where W is the number of T_train windows
    """
    with torch.no_grad():

        if distribution == 'Normal':
            # get k-step y_true of shape K x k_step x dy x W
            y_true = y_test[:, T_train:].unfold(1, k_step, 1).permute(0, 3, 2, 1)

        elif distribution == 'Binomial':
            y_true = logrates[:, T_train:].unfold(1, k_step, 1).permute(0, 3, 2, 1)

        if align is not None:
            y_test = align.f_enc(y_test)
            decoder = align.f_dec
        else:
            decoder = vae.decoder

        # form windows of size T_train to get y of shape K x T_train x dy x W
        y_in = y_test[:, :-k_step, :].unfold(1, T_train, 1).permute(0, 3, 2, 1)

        r2_ys = []

        # iterate over W windows
        for i in range(y_in.shape[-1]):

            # get x_{t+T_train} from y_{t:t+T_train}
            x = vae.encoder.sample(y_in[..., i])[0]

            # sample k_steps from x_{t+T_train}
            x_k_ahead, _, _ = vae.prior.sample_k_step_ahead(x[:, -1, :][:, np.newaxis, :], k_step, True)

            y_pred = decoder.compute_param(torch.hstack(x_k_ahead)) # shape K x k_step x dy
            if distribution == 'Normal':
                y_pred = y_pred[0]

            r2_ys.append(compute_r2(y_pred, y_true[..., i]))

    return torch.stack(r2_ys)


def plot_k_step_pred(r2_y, fig=None, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    k_step = r2_y.shape[1]

    ax.plot(np.arange(1, k_step + 1), torch.mean(r2_y, (0, 2)), 'o')

    return ax