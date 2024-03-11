import torch
import torch.nn as nn

from utils import vectorize, tensorize
from priors import *


def sample_k_step_ahead(prior, x, K, u=None, keep_trajectory=False):
    x_trajectory, means, vars = [], [], []

    mu, var = prior.compute_param(x)

    means.append(mu)
    vars.append(var)
    x_trajectory.append(mu + torch.sqrt(var) * torch.randn(mu.shape, device=x.device))

    for t in torch.arange(1, K):
        if u is not None:
            mu, var = prior.compute_param(torch.cat((x_trajectory[-1], u[:, t-1].unsqueeze(1)), -1))
        else:
            mu, var = prior.compute_param(x_trajectory[-1])

        means.append(mu)
        vars.append(var)
        x_trajectory.append(mu + torch.sqrt(var) * torch.randn(mu.shape, device=x.device))

    if keep_trajectory:
        return torch.hstack(x_trajectory), torch.hstack(means), torch.vstack(vars)
    else:
        return x_trajectory[-1], means[-1], vars[-1]


def compute_k_step_log_q(x_samples, mu, var, k_step, method='k_step'):

    if method == 'multi_step':
        x_k = vectorize(x_samples[:, 1:], k_step)
        mu_k = vectorize(mu[:, 1:], k_step)
        var_k_step = vectorize(var[:, 1:], k_step)

        "mean across k steps"
        log_q = torch.mean(Normal(mu_k, torch.sqrt(var_k_step)).log_prob(x_k), -2, keepdim=True)
        log_q = tensorize(log_q, mu.shape[0])

    else:
        log_q = Normal(mu[:, k_step:], torch.sqrt(var[:, k_step:])).log_prob(x_samples[:, k_step:])

        "sum across time and dx"
    log_q = torch.sum(log_q, (-1, -2))

    for i in torch.arange(k_step - 1):
        k_ahead = k_step - i - 1

        if method == 'multi_step':
            temp = Normal(mu[..., -k_ahead:, :], torch.sqrt(var)[..., -k_ahead:, :]).log_prob(
                    x_samples[..., -k_ahead:, :])/k_ahead
            temp = temp/k_ahead
        else:
            temp = Normal(mu[:, -1, :], torch.sqrt(var)[:, -1, :]).log_prob(
                x_samples[:, -1, :]).unsqueeze(1)

        log_q = log_q + torch.sum(temp, (-1, -2))

    return log_q


def compute_k_step_prior(x_samples, k_step, prior, method='k_step'):
    """
    :param x_samples: samples from variational posterior of size B x T x dx + dx
    :param k_step: number of steps
    :param dx: latent dimension
    :param method: either 'k_step' or 'multi_step'; 'multi_step computes KL over all k steps'

    :returns log_prior
    """
    dx = prior.dx
    keep_trajectory = True if method == 'multi_step' else False

    x_k_step = vectorize(x_samples[..., 1:, :], k_step)
    _, mu_k_ahead, var_k_ahead = sample_k_step_ahead(prior, x_samples[:, : -k_step].reshape(-1, 1, x_samples.shape[-1]),
                                                     k_step, u=x_k_step[..., dx:], keep_trajectory=keep_trajectory)

    if method == 'multi_step':
        log_k_step_prior = torch.mean(Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_k_step[..., :dx]), -2, keepdim=True)
    else:
        log_k_step_prior = Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_k_step[:, -1, :dx].unsqueeze(1))

    log_k_step_prior = torch.sum(tensorize(log_k_step_prior, x_samples.shape[0]), (-1, -2))

    # compute prediction for the last k-1 steps
    for i in torch.arange(k_step - 1):
        k_ahead = k_step - i - 1

        _, mu_k, var_k = sample_k_step_ahead(prior, x_samples[..., -(k_ahead + 1), :].unsqueeze(1), k_ahead,
                                             u=x_samples[..., -k_ahead:, dx:], keep_trajectory=keep_trajectory)

        if method == 'multi_step':
            log_prior = Normal(mu_k, torch.sqrt(var_k)).log_prob(x_samples[:, -k_ahead:].reshape(-1, k_ahead, dx))
            log_prior = log_prior/k_ahead
        else:
            log_prior = Normal(mu_k, torch.sqrt(var_k)).log_prob(x_samples[:, -1, :dx].unsqueeze(1))

        log_k_step_prior = log_k_step_prior + torch.sum(log_prior, (-1, -2))

    return log_k_step_prior


