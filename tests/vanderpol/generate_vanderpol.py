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


def noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=1.5, dt=1e-2, noise_type='poisson'):
    x = np.empty((K, T, 2))
    y = np.empty((K, T, dy))
    # y = torch.empty((K, T, dy))

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
    # elif noise_type == 'bernoulli':
    #     log_rates = x[:, 0] @ C + b
    #     y[:, 0] = npr.binomial(1, sigmoid(log_rates))

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

    if noise_type == 'bernoulli':
        xs = np.vstack([x[k] for k in range(K)])
        mu = np.mean(xs, 0, keepdims=True)
        sigma = np.std(xs, 0, keepdims=True)

        x_normalized = (x - mu) / sigma  # should broadcast correctly

        log_rates = x_normalized @ C + b
        y = npr.binomial(1, sigmoid(log_rates))
    return x, y, C, b

sigma_x = 0.5  # state noise
sigma_y = 0.1  # observation noise
dt = 1e-2  # euler integration time step
mu = 1.5
K = 1_000  # number of batches
T = 300  # length of time series

"different number of observations for sessions/animals"
if noise_type == 'gaussian':
    dys = [30, 30, 40, 50]
elif noise_type == 'bernoulli' or noise_type == 'poisson':
    dys = [100, 100, 150, 200]

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)


data_all = []

for dy in tqdm(dys):
    x, y, C, b = noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=mu, dt=dt, noise_type=noise_type)
    data = {}
    data['x'] = torch.from_numpy(x).float()
    data['y'] = torch.from_numpy(y).float()
    data['C'] = torch.from_numpy(C).float()
    data['b'] = torch.from_numpy(b).float()

    data_all.append(data)

# In[]
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
#
# # for k in range(K):
# #     axs[0].plot(data_all[0]['x'][k, :, 0], data_all[0]['x'][k, :, 1], alpha=0.3)
# #     for d in range(30):
# #         axs[1].plot(data_all[0]['y'][0, :, d], alpha=0.3)
# # fig.show()


data_path = 'data'
print(torch.sum(torch.isnan(torch.from_numpy(x))),
      torch.sum(torch.isnan(torch.from_numpy(y))),
      torch.max(torch.from_numpy(y)))
if not os.path.isdir(data_path):
    os.makedirs(data_path)

torch.save(data_all, f'data/noisy_vanderpol_{noise_type}.pt')
