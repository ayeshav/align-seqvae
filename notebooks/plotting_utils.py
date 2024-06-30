import torch
import numpy as np
from tqdm import tqdm


def compute_vector_field(dynamics, xmin, xmax, ymin, ymax, device='cpu'):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    :return: a N by dy matrix
    """
    xt, yt = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    queries = np.stack([xt.ravel(), yt.ravel()]).T
    queries = torch.from_numpy(queries).float().to(device)
    with torch.no_grad():
        N = queries.shape[0]
        vel = torch.zeros(queries.shape)
        for n in tqdm(range(N)):
            vel[n, :] = (dynamics(queries[[n]]) - queries[[n]]).to('cpu')

    vel_x = vel[:, 0].reshape(xt.shape[0], xt.shape[1])
    vel_y = vel[:, 1].reshape(yt.shape[0], yt.shape[1])
    speed = torch.sqrt(vel_x ** 2 + vel_y ** 2)
    return xt, yt, vel_x, vel_y, speed
