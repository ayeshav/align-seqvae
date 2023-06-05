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


mu = 1.5
dt = 1e-1

"different number of observations for sessions/animals"
N = [20, 50, 30]

"let's start with the same number of trials and trial length"
K = 400
T = 1000

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)

"let's generate data for the first session and transform it for the rest"
x = np.empty((K, T, dx))

data_all = []

for j in range(len(N)):

    data = {}

    x0 = np.random.uniform(-1,1, (K, dx))

    C = np.random.randn(dx, N[j]) / np.sqrt(dx)
    y = np.empty((K, T, N[j]))

    for i in range(K):
        x[i] = vanderpol_osc(T, t_eval, mu, x0[i])[1:]
        y[i] = x[i]@C

    data['x'] = torch.from_numpy(x.transpose(1,0,2))
    data['y'] = torch.from_numpy(y.transpose(1,0,2))
    data['C'] = C

    data_all.append(data)

torch.save(data_all, 'vanderpol.pt')





