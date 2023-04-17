import numpy as np
import torch
from scipy.integrate import odeint


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
N = [200, 50, 100]

"let's start with the same number of trials and trial length"
K = 400
T = 1000

dx = 2
t_eval = np.arange(0, (T+1) * dt, dt)

"let's generate data for the first session and transform it for the rest"
x0 = 20 * np.random.rand(K, dx) - 5
x = np.empty((K, T, dx))

data = {}

for j in range(len(N)):
    C = np.random.randn(dx, N[j]) / np.sqrt(dx)
    y = np.empty((K, T, N[j]))

    for i in range(K):
        x[i] = vanderpol_osc(T, t_eval, mu, x0[i])[1:]
        y[i] = x[i]@C

    data[str(j)] = (torch.from_numpy(x.transpose(1,0,2)), torch.from_numpy(y.transpose(1,0,2)))

torch.save(data, 'vanderpol.pt')
i = 0





