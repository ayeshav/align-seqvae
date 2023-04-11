import numpy as np
from scipy.integrate import odeint


def vanderpol_osc(T, t_eval, mu, x0):

    def f(state,t):
        x, y = state

        xdot = y
        ydot = -mu*(x**2-1)*y - x

        return xdot, ydot

    result = odeint(f, x0, t_eval)

    return result
