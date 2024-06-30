import sys
sys.path.append('../')

import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
from utils import *

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)

noise_type = 'binomial'

sigma_y = 0.1  # observation noise
dt = 0.1  # euler integration time step
K = 2_000  # number of batches
T = 300  # length of time series

N_max = 4

# parameters for noisy vdp
sigma_x = 0.5  # state noise
mu = 1.5
dx = 2

"different number of observations for sessions/animals"
if noise_type == 'gaussian':
    dys = [40, 35, 55]
elif noise_type == 'binomial' or noise_type == 'poisson':
    dys = [250, 250, 200, 300]

t_eval = np.arange(0, (T + 1) * dt, dt)

data_all = []

for dy in tqdm(dys):
    x = sim_noisy_vanderpol(K, T, sigma_x, mu=mu, dt=dt)

    if noise_type == 'binomial':
        C = 4 * npr.rand(dx, dy) - 2
    else:
        C = npr.randn(dx, dy) / np.sqrt(dx)

    b = np.zeros((1, dy))

    y, x = get_observations(x, C, b, sigma_y, noise_type=noise_type)

    data = {'x': torch.from_numpy(x).float(),
            'y': torch.from_numpy(y).float(),
            'C': torch.from_numpy(C).float(),
            'b': torch.from_numpy(b).float()}

    data_all.append(data)

    print(torch.sum(torch.isnan(torch.from_numpy(x))),
          torch.sum(torch.isnan(torch.from_numpy(y))),
          torch.max(torch.from_numpy(y)))

torch.save(data_all, f'noisy_vanderpol_{noise_type}_dt={dt}.pt')
