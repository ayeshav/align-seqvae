import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
from ..utils import *

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)

noise_type = 'gaussian'


sigma_y = 0.1  # observation noise
dt = 0.01  # euler integration time step
K = 2_000  # number of batches
T = 500  # length of time series
n_init = 1_000

N_max = 4

# parameters for lorenz
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

sigma_x = 0
dx = 3

"different number of observations for sessions/animals"
if noise_type == 'gaussian':
    dys = [40, 35, 55]
elif noise_type == 'bernoulli' or noise_type == 'poisson':
    dys = [250, 250, 200, 300]

t_eval = np.arange(0, (T + n_init + 1) * dt, dt)

data_all = []

for dy in tqdm(dys):
    x = sim_lorenz(K, T + n_init, dt)

    C = (npr.rand(dx, dy) + 1) * np.sign(npr.randn(dx, dy))
    b = np.zeros(1, dy)

    y, x_norm = get_observations(x[:, n_init:, :], C, b, sigma_y, noise_type=noise_type, N_max=N_max,
                                 norm_latent=True)

    data = {'x': torch.from_numpy(x).float(),
            'y': torch.from_numpy(y).float(),
            'C': torch.from_numpy(C).float(),
            'b': torch.from_numpy(b).float()}

    data_all.append(data)

    print(torch.sum(torch.isnan(torch.from_numpy(x))),
          torch.sum(torch.isnan(torch.from_numpy(y))),
          torch.max(torch.from_numpy(y)))

torch.save(data_all, f'lorenz_{noise_type}_dt={dt}.pt')
