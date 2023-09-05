import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import compute_wasserstein, vectorize_x

Softplus = torch.nn.Softplus()
eps = 1e-6


class SeqVae(nn.Module):
    def __init__(self, prior, encoder, decoder, device='cpu', k_step=1):
        super().__init__()
        self.k_step = k_step

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _prior(self, x_samples):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        if self.k_step == 1:
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                K_ahead = min(self.k_step, x_samples[:, t + 1:].shape[1])
                _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                log_prior = log_prior + torch.sum(
                    Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def forward(self, y, beta=1.):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return -elbo


class CondSeqVae(nn.Module):
    def __init__(self, prior, encoder, decoder, readout_behav=None, device='cpu', k_step=1):
        super().__init__()
        self.k_step = k_step

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.readout_behav = readout_behav

        self.device = device

    def _compute_k_step_log_q(self, y):
        x_samples, mu, var = self.encoder.sample(y)

        if self.k_step == 1:
            log_q = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x_samples), (-2, -1))
        else:

            x_k = vectorize_x(x_samples[:, 1:], self.k_step)
            mu_k = vectorize_x(mu[:, 1:], self.k_step)
            var_k_step = vectorize_x(var[:, 1:], self.k_step)

            "mean across k steps"
            log_q = torch.mean(Normal(mu_k, torch.sqrt(var_k_step)).log_prob(x_k), -2)

            "sum across time and dx"
            log_q = torch.sum(log_q.reshape(mu.shape[0], -1, mu.shape[2]), (-1, -2))

            for i in torch.arange(self.k_step - 1):
                K_ahead = self.k_step - i - 1
                temp = torch.sum(Normal(mu[..., -K_ahead:, :], torch.sqrt(var)[..., -K_ahead:, :]).log_prob(
                    x_samples[..., -K_ahead:, :]), (-1, -2))
                log_q = log_q + temp / K_ahead

        return x_samples, mu, var, log_q

    def _compute_k_step_log_prior(self, x_samples, u):
        """
        function to compute multi-step prediction (1/K) * sum_{i=1:K} p(x_{t+k} | x_t)
        """
        dx = x_samples.shape[-1]
        du = u.shape[-1]

        x_prev = torch.cat((x_samples[:, :-self.k_step], u[:, :-self.k_step]), axis=-1)

        if self.k_step == 1:
            log_k_step_prior = self.prior(x_prev, x_samples[..., 1:, :])

        else:
            # get vectorized k-step windows of x_samples BW x k_step x dx
            x_k = vectorize_x(x_samples[..., 1:, :], self.k_step)
            u_k = vectorize_x(u[..., 1:, :], self.k_step)

            _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_prev.reshape(-1, 1, dx + du),
                                                                        self.k_step, u=u_k,
                                                                        keep_trajectory=True)
            # take mean along k-steps
            log_k_step_prior = torch.mean(Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_k), -2)

            # sum along time and dx
            log_k_step_prior = torch.sum(log_k_step_prior.reshape(x_samples.shape[-3], -1, dx), (-1, -2))

            # compute prediction for the last k-1 steps
            for i in torch.arange(self.k_step - 1):
                K_ahead = self.k_step - i - 1

                x_prev = torch.cat((x_samples[..., -(K_ahead + 1), :], u[..., -(-K_ahead + 1), :]), axis=-1)
                _, mu_k, var_k = self.prior.sample_k_step_ahead(x_prev.unsqueeze(1), K_ahead, u=u[..., -K_ahead:, :],
                                                                keep_trajectory=True)
                log_prior = torch.sum(
                    Normal(mu_k, torch.sqrt(var_k)).log_prob(x_samples[:, -K_ahead:].reshape(-1, K_ahead, dx)),
                    (-1, -2))
                log_k_step_prior = log_k_step_prior + log_prior / K_ahead

        return log_k_step_prior

    def forward(self, y, u, y_behav=None, beta=1.):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :param u: U is a tensor of input of size B x T x Du
        :param y_behav: tensor of behavior (example, hand velocity) of size B x T x Y_behav
        :return:
        """
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self._compute_k_step_log_q(torch.cat((y, u), -1))

        log_prior_0 = torch.sum(Normal(torch.zeros(1, device=self.device),
                                       0.5 * torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
        log_q_0 = torch.sum(Normal(mu[:, 0], torch.sqrt(var[:, 0])).log_prob(x_samples[:, 0]), -1)

        kl_0 = log_prior_0 - log_q_0

        # given samples, compute the log prior
        log_prior = self._compute_k_step_log_prior(x_samples, u)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (kl_0 + log_prior - log_q))

        if y_behav is not None:
            log_like_behav = self.readout_behav(x_samples, y_behav)

            return -elbo - torch.mean(log_like_behav)
        else:
            return -elbo


class DualAnimalSeqVae(nn.Module):
    def __init__(self, prior, encoder, decoder, align,
                 device='cpu', k_step=1):
        super().__init__()
        self.k_step = k_step

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.align = align
        self.device = device

    def _prior(self, x_samples):
        """
        Compute log p(x_t | x_{t-1})
        :param x_samples: A tensor of latent samples of dimension Batch by Time by Dy
        :return:
        """
        if self.k_step == 1:
            log_prior = torch.sum(Normal(torch.zeros(1, device=self.device),
                                         torch.ones(1, device=self.device)).log_prob(x_samples[:, 0]), -1)
            log_prior = log_prior + self.prior(x_samples[:, :-1], x_samples[:, 1:])
        else:
            log_prior = 0

            for t in range(x_samples.shape[1] - 1):
                K_ahead = min(self.k_step, x_samples[:, t + 1:].shape[1])
                _, mu_k_ahead, var_k_ahead = self.prior.sample_k_step_ahead(x_samples[:, t], K_ahead)
                log_prior = log_prior + torch.sum(
                    Normal(mu_k_ahead, torch.sqrt(var_k_ahead)).log_prob(x_samples[:, t + K_ahead]), -1)
        return log_prior

    def ref_animal_loss(self, y_ref, beta=1.):
        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y_ref)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # given samples, compute the log likelihood
        log_like = self.decoder(x_samples, y_ref)

        # compute the elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return x_samples, -elbo

    def other_animal_loss(self, y_other, beta=1.):
        # first pass data from f_enc
        y_hat = self.align.f_enc(y_other)

        # pass data through encoder and get mean, variance, samples and log density
        x_samples, mu, var, log_q = self.encoder(y_hat)

        # given samples, compute the log prior
        log_prior = self._prior(x_samples)

        # can we reconstruct y_hat using regular decoder

        # log_like = self.decoder(x_samples, y_hat)
        #
        # # compute the likelihood using animal specific likelihood
        # log_like = log_like + self.other_animal_decoder(x_samples, y_other)

        # for now use the f_dec directly
        log_like = self.align.f_dec(x_samples, y_other)

        # compute elbo
        elbo = torch.mean(log_like + beta * (log_prior - log_q))
        return x_samples, -elbo

    def compute_regularizer(self, x_ref_samples, x_other_samples, reg_weight, velocity=False):

        dx = x_ref_samples.shape[-1]

        mu_ref = torch.mean(x_ref_samples.view(-1, dx), 0)
        cov_ref = torch.cov(x_ref_samples.view(-1, dx).T)

        mu_other = torch.mean(x_other_samples.view(-1, dx), 0)
        cov_other = torch.cov(x_other_samples.view(-1, dx).T)

        # compute Wassertein
        reg_shape = compute_wasserstein(mu_ref, cov_ref,
                                        mu_other, cov_other)

        if velocity:
            vel_ref = self.prior.compute_param(x_ref_samples)[0] - x_ref_samples
            vel_other = self.prior.compute_param(x_other_samples)[0] - x_other_samples

            mu_ref = torch.mean(vel_ref.view(-1, dx), 0)
            cov_ref = torch.cov(vel_ref.view(-1, dx).T)

            mu_other = torch.mean(vel_other.view(-1, dx), 0)
            cov_other = torch.cov(vel_other.view(-1, dx).T)

            reg_vel = compute_wasserstein(mu_ref, cov_ref,
                                          mu_other, cov_other)

            return reg_weight[0] * reg_shape + reg_weight[1] * reg_vel
        else:
            return reg_weight[0] * reg_shape

    def forward(self, y_ref, y_other, beta=1., align_mode=False, reg_weight=(100,100), velocity=False):
        """
        In the forward method, we compute the negative elbo and return it back
        :param y: Y is a tensor of observations of size Batch by Time by Dy
        :return:
        """
        # ref animal elbo
        x_ref_samples, ref_neg_elbo = self.ref_animal_loss(y_ref, beta=beta)

        # other animal elbo
        x_other_samples, other_neg_elbo = self.other_animal_loss(y_other, beta=beta)

        reg = self.compute_regularizer(x_ref_samples, x_other_samples, reg_weight, velocity=velocity)

        if not align_mode:
            return ref_neg_elbo + other_neg_elbo + reg
        else:
            return other_neg_elbo