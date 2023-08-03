import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sim_dynamics import *

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)

noise_type = 'bernoulli'
dynamics = 'lorenz'

sigmoid = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log(1 + np.exp(z))


def get_observations(x, dy, sigma_y, noise_type='bernoulli', N_max=4):
    K, T, dx = x.shape
    y = np.empty((K, T, dy))

    # generate random readout
    C = npr.randn(dx, dy) / np.sqrt(dx)
    b = 2 * npr.rand(1, dy) - 1

    if noise_type == 'gaussian':
        y = x @ C + sigma_y * npr.randn(K, T, dy)
    elif noise_type == 'poisson':
        log_rates = x @ C + b
        y = npr.poisson(softplus(log_rates))
    elif noise_type == 'bernoulli':
        C = 4 * npr.rand(2, dy) - 2
        # b = 0.5 * npr.rand(1, dy) - 0.25
        b = np.zeros((1, dy))

        log_rates = x @ C + b
        y = npr.binomial(N_max, sigmoid(log_rates))

    return y, C, b


sigma_y = 0.1  # observation noise
dt = 1e-1  # euler integration time step
K = 2_000  # number of batches
T = 300  # length of time series
N_max = 4

# parameters for noisy vdp
sigma_x = 0.5  # state noise
mu = 1.5

# parameters for lorenz
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0


"different number of observations for sessions/animals"
if noise_type == 'gaussian':
    dys = [30, 30, 40, 50]
elif noise_type == 'bernoulli' or noise_type == 'poisson':
    dys = [250, 250, 200, 300]

t_eval = np.arange(0, (T+1) * dt, dt)

data_all = []

for dy in tqdm(dys):
    # x = sim_noisy_vanderpol(K, T, sigma_x, mu=mu, dt=dt)
    x = sim_lorenz(K, T, dt)
    y, C, b = get_observations(x, dy, sigma_y, noise_type=noise_type, N_max=N_max)

    data = {}
    data['x'] = torch.from_numpy(x).float()
    data['y'] = torch.from_numpy(y).float()
    data['C'] = torch.from_numpy(C).float()
    data['b'] = torch.from_numpy(b).float()

    data_all.append(data)

    print(torch.sum(torch.isnan(torch.from_numpy(x))),
          torch.sum(torch.isnan(torch.from_numpy(y))),
          torch.max(torch.from_numpy(y)))

# torch.save(data_all, f'noisy_vanderpol_{noise_type}_dt={dt}.pt')
