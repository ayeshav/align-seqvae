import numpy as np
import numpy.random as npr
import torch
import os
from tqdm import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)

noise_type = 'bernoulli'
sigmoid = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log(1 + np.exp(z))


def noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=1.5, dt=1e-2, noise_type='bernoulli', N_max=4):
    x = np.empty((K, T, 2))
    y = np.empty((K, T, dy))

    # generate random readout
    C = npr.randn(2, dy) / np.sqrt(2)
    b = 2 * npr.rand(1, dy) - 1

    # generate initial conditions
    x[:, 0] = 6 * npr.rand(K, 2) - 3

    if noise_type == 'gaussian':
        y[:, 0] = x[:, 0] @ C + sigma_y * npr.randn(K, dy)
    elif noise_type == 'poisson':
        log_rates = x[:, 0] @ C + b
        y[:, 0] = npr.poisson(softplus(log_rates))
    elif noise_type == 'bernoulli':
        C = 4 * npr.rand(2, dy) - 2
        # b = 0.5 * npr.rand(1, dy) - 0.25
        b = np.zeros((1, dy))

        log_rates = x[:, 0] @ C + b
        y[:, 0] = npr.binomial(N_max, sigmoid(log_rates))

    # propagate time series
    for t in range(1, T):
        vel = np.empty((K, 2))
        vel[:, 0] = mu * (x[:, t - 1, 0] - x[:, t - 1, 0] ** 3 / 3 - x[:, t - 1, 1])
        vel[:, 1] = x[:, t - 1, 0] / mu

        x[:, t] = x[:, t - 1] + dt * (vel + sigma_x * npr.randn(K, 2))

        if noise_type == 'gaussian':
            y[:, t] = x[:, t] @ C + sigma_y * npr.randn(K, dy)
        elif noise_type == 'poisson':
            log_rates = x[:, t] @ C + b
            y[:, t] = npr.poisson(softplus(log_rates))
        elif noise_type == 'bernoulli':
            log_rates = x[:, t] @ C + b
            y[:, t] = npr.binomial(N_max, sigmoid(log_rates))

    return x, y, C, b

sigma_x = 0.5  # state noise
sigma_y = 0.1  # observation noise
dt = 1e-1  # euler integration time step
mu = 1.5
K = 2_000  # number of batches
T = 300  # length of time series
N_max = 4

"different number of observations for sessions/animals"
if noise_type == 'gaussian':
    dys = [30, 30, 40, 50]
elif noise_type == 'bernoulli' or noise_type == 'poisson':
    dys = [250, 250, 200, 300]

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)


data_all = []

for dy in tqdm(dys):
    x, y, C, b = noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y,
                                    mu=mu, dt=dt, noise_type=noise_type, N_max=N_max)
    data = {}
    data['x'] = torch.from_numpy(x).float()
    data['y'] = torch.from_numpy(y).float()
    data['C'] = torch.from_numpy(C).float()
    data['b'] = torch.from_numpy(b).float()

    data_all.append(data)


data_path = 'data'
print(torch.sum(torch.isnan(torch.from_numpy(x))),
      torch.sum(torch.isnan(torch.from_numpy(y))),
      torch.max(torch.from_numpy(y)))
if not os.path.isdir(data_path):
    os.makedirs(data_path)

torch.save(data_all, f'data/noisy_vanderpol_{noise_type}_dt={dt}.pt')
