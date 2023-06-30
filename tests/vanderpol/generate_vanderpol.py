import numpy as np
import numpy.random as npr
import torch
import os
from scipy.integrate import odeint
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.random.manual_seed(42)
npr.seed(0)


def vanderpol_osc(T, t_eval, mu, x0):
    "function to generate noiseless vanderpol oscillator"

    def f(state,t):
        x, y = state

        xdot = y
        ydot = -mu*(x**2-1)*y - x

        return xdot, ydot

    result = odeint(f, x0, t_eval)

    return result


def noisy_vanderpol(T, t_eval, params, x0):
    dt = t_eval[1] - t_eval[0]
    mu = params['mu']
    tau_1 = params['tau_1']
    tau_2 = params['tau_2']
    sigma = params['sigma']
    scale = params['scale']

    x = np.zeros((T, 2))
    x[0, 0] = x0[0]
    x[0, 1] = x0[1]

    for dx in range(1, T):
        x_next = x[dx - 1, 0] + (1 / scale) * (dt / tau_1) * scale * x[dx - 1, 1]
        y_next = x[dx - 1, 1] + (1 / scale) * (dt / tau_2) * (mu * (1 - scale**2 * x[dx - 1, 0]**2) * scale * x[dx - 1, 1] - scale * x[dx - 1, 0])

        x[dx, 0] = x_next + (sigma / scale) * np.random.randn()
        x[dx, 1] = y_next + (sigma / scale) * np.random.randn()

    return x


def noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=1.5, dt=1e-2):
    x = np.empty((K, T, 2))
    y = np.empty((K, T, dy))

    # generate random readout
    C = npr.randn(2, dy) / np.sqrt(2)

    # generate initial conditions
    x[:, 0] = 6 * npr.rand(K, 2) - 3
    y[:, 0] = x[:, 0] @ C + sigma_y * npr.randn(K, dy)

    # propagate time series
    for t in range(1, T):
        vel = np.empty((K, 2))
        vel[:, 0] = mu * (x[:, t -1, 0] - x[:, t - 1, 0] ** 3 / 3 - x[:, t - 1, 1])
        vel[:, 1] = x[:, t - 1, 0] / mu

        x[:, t] = x[:, t - 1] + dt * (vel + sigma_x * npr.randn(K, 2))
        y[:, t] = x[:, t] @ C + sigma_y * npr.randn(K, dy)
    return x, y, C


# "params for vdp"
# mu = 1.5
# dt = 1e-2
#
# "params dict for noisy vdp"
# params = {}
# params['mu'] = mu
# params['tau_1'] = 0.1
# params['tau_2'] = 0.1
# params['sigma'] = 0.05  # noise add into euler integration
# params['scale'] = 1 / 0.4

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
    x, y, C = noisy_vanderpol_v2(K, T, dy, sigma_x, sigma_y, mu=mu, dt=dt)
    data =  {}
    data['x'] = torch.from_numpy(x)
    data['y'] = torch.from_numpy(y)
    data['C'] = C

    data_all.append(data)

# In[]
fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

for k in range(K):
    axs[0].plot(data_all[0]['x'][k, :, 0], data_all[0]['x'][k, :, 1], alpha=0.3)
    for d in range(30):
        axs[1].plot(data_all[0]['y'][0, :, d], alpha=0.3)
fig.show()


# x = np.empty((K, T, dx))
# for j in range(len(N)):
#
#     data = {}
#
#     x0 = 6 * npr.rand(K, dx) - 3
#
#     C = np.random.randn(dx, N[j]) / np.sqrt(dx)
#     y = np.empty((K, T, N[j]))
#
#     for i in range(K):
#         x[i] = noisy_vanderpol(T+1, t_eval, params, x0[i])[1:]
#         y[i] = x[i]@C + np.expand_dims(np.random.randn(N[j]),0)*Q
#
#     data['x'] = torch.from_numpy(x)
#     data['y'] = torch.from_numpy(y)
#     data['C'] = C
#
#     data_all.append(data)

data_path = 'data'
print(torch.sum(torch.isnan(torch.from_numpy(y))))
if not os.path.isdir(data_path):
    os.makedirs(data_path)

torch.save(data_all, 'data/noisy_vanderpol.pt')
