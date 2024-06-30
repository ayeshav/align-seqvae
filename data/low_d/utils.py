import numpy as np
import numpy.random as npr
import torch
from scipy.integrate import odeint

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)

sigmoid = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log(1 + np.exp(z))


def sim_noisy_vanderpol(K, T, sigma_x, mu=1.5, dt=1e-2):
    x = np.empty((K, T, 2))

    # generate initial conditions
    x[:, 0] = 6 * npr.rand(K, 2) - 3

    # propagate time series
    for t in range(1, T):
        vel = np.empty((K, 2))
        vel[:, 0] = mu * (x[:, t - 1, 0] - x[:, t - 1, 0] ** 3 / 3 - x[:, t - 1, 1])
        vel[:, 1] = x[:, t - 1, 0] / mu

        x[:, t] = x[:, t - 1] + dt * (vel + sigma_x * npr.randn(K, 2))

    return x


def sim_lorenz(K, T, dt=1e-2, rho=28.0, sigma=10.0, beta=8.0/3.0):
    """
    :param K: number of trajectories
    :param T: length of each trajectory
    :return:
    """
    dx = 3
    x = np.zeros((K, T, dx))

    times = np.arange(0.0, (T + 1) * dt, dt)

    def f(state, t):  # ode for lorenz
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    initial_states = 30 * npr.rand(K, dx) - 15
    for n in range(K):
        states = odeint(f, initial_states[n], times)
        x[n, :, :] = states[1:]  # don't care about initial conditions

    return x


def get_observations(x, C, b, sigma_y, noise_type='binomial', norm_latent=False, N_max=4):
    K, T, dx = x.shape
    dy = C.shape[1]

    y = np.empty((K, T, dy))

    if norm_latent:
        mu = np.mean(x.reshape(-1, dx))[np.newaxis]
        sigma = np.std(x.reshape(-1, dx))[np.newaxis]
        x = (x - mu) / sigma

    if noise_type == 'gaussian':
        y = x @ C + sigma_y * npr.randn(K, T, dy)
    elif noise_type == 'poisson':
        log_rates = x @ C + b
        y = npr.poisson(softplus(log_rates))
    elif noise_type == 'binomial':
        log_rates = x @ C + b
        y = npr.binomial(N_max, sigmoid(log_rates))

    return y, x