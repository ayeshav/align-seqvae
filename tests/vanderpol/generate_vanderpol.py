import numpy as np
import torch
from scipy.integrate import odeint

torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(0)


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


"params for vdp"
mu = 1.5
dt = 1e-2

"params dict for noisy vdp"
params = {}
params['mu'] = mu
params['tau_1'] = 0.1
params['tau_2'] = 0.1
params['sigma'] = 0.05  # noise add into euler integration
params['scale'] = 1 / 0.4

Q = 0.01 # observation noise

"different number of observations for sessions/animals"
N = [20, 50, 30]

"let's start with the same number of trials and trial length"
K = 400
T = 1000

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)

x = np.empty((K, T, dx))

data_all = []

for j in range(len(N)):

    data = {}

    x0 = np.random.uniform(-1,1, (K, dx))

    C = np.random.randn(dx, N[j]) / np.sqrt(dx)
    y = np.empty((K, T, N[j]))

    for i in range(K):
        x[i] = noisy_vanderpol(T+1, t_eval, params, x0[i])[1:]
        y[i] = x[i]@C + np.expand_dims(np.random.randn(N[j]),0)*Q

    data['x'] = torch.from_numpy(x.transpose(1,0,2))
    data['y'] = torch.from_numpy(y.transpose(1,0,2))
    data['C'] = C

    data_all.append(data)

torch.save(data_all, 'noisy_vanderpol.pt')





