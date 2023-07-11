import numpy as np
import numpy.random as npr
import torch
import os
from scipy.integrate import odeint
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)


def noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=1.5, dt=1e-2, noise_type='poisson'):
    x = np.empty((K, T, 2))
    y = torch.empty((K, T, dy))

    # generate random readout
    C = npr.randn(2, dy) / np.sqrt(2)

    C = torch.randn((dy, 2), dtype=torch.float64)
    C = (1 / np.sqrt(2)) * (C / torch.norm(C, dim=1).unsqueeze(1))

    b = torch.log(5 + 10 * torch.rand(dy, dtype=torch.float64))

    # generate initial conditions
    x[:, 0] = 6 * npr.rand(K, 2) - 3

    if noise_type == 'gaussian':
        y[:, 0] = x[:, 0] @ C.T + sigma_y * npr.randn(K, dy)
    elif noise_type == 'poisson':
        y[:, 0] = torch.poisson(dt * torch.exp(torch.from_numpy(x[:, 0]) @ C.T + b.unsqueeze(0)))

    # propagate time series
    for t in range(1, T):
        vel = np.empty((K, 2))
        vel[:, 0] = mu * (x[:, t - 1, 0] - x[:, t - 1, 0] ** 3 / 3 - x[:, t - 1, 1])
        vel[:, 1] = x[:, t - 1, 0] / mu

        x[:, t] = x[:, t - 1] + dt * (vel + sigma_x * npr.randn(K, 2))

        if noise_type == 'gaussian':
            y[:, t] = x[:, t] @ C.T + sigma_y * npr.randn(K, dy)
        elif noise_type == 'poisson':
            y[:, t] = torch.poisson(dt * torch.exp(torch.from_numpy(x[:, t]) @ C.T + b.unsqueeze(0)))
    return x, y, C, b

sigma_x = 0.5  # state noise
sigma_y = 0.1  # observation noise
dt = 1e-2  # euler integration time step
mu = 1.5
K = 1_000  # number of batches
T = 300  # length of time series

"different number of observations for sessions/animals"
dys = [30, 30, 40, 50]

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)


data_all = []

for dy in dys:
    x, y, C, b = noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=mu, dt=dt)
    data = {}
    data['x'] = torch.from_numpy(x)
    data['y'] = y
    data['C'] = C
    data['b'] = b

    data_all.append(data)

# In[]
fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

for k in range(K):
    axs[0].plot(data_all[0]['x'][k, :, 0], data_all[0]['x'][k, :, 1], alpha=0.3)
    for d in range(30):
        axs[1].plot(data_all[0]['y'][0, :, d], alpha=0.3)
fig.show()



data_path = 'data'
print(torch.sum(torch.isnan(y)))
if not os.path.isdir(data_path):
    os.makedirs(data_path)

torch.save(data_all, 'data/noisy_vanderpol_poisson.pt')
