import abc
import torch
import numpy as np


class SDE(abc.ABC):
    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, feature, x, flags, t, is_adj=True):
                drift, diffusion = sde_fn(x, t) if is_adj else sde_fn(feature, t)
                score = score_fn(feature, x, flags, t)
                drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, feature, x, flags, t, is_adj=True):
                f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
                score = score_fn(feature, x, flags, t)
                rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.01, beta_max=0.1, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        z = torch.randn(*shape)
        b = self.beta_1
        a = self.beta_0
        return (b - a) / 2 * z + (a + b) / 2

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1, -2)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
        return logps

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None] * x - x
        G = sqrt_beta
        return f, G

    def transition(self, x, t, dt):
        log_mean_coeff = 0.25 * dt * (2 * self.beta_0 + (2 * t + dt) * (self.beta_1 - self.beta_0))
        mean = torch.exp(-log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        x = x + x.transpose(-1, -2)
        return x

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

    def transition(self, x, t, dt):
        std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
              torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt))
        std = torch.sqrt(std)
        mean = x
        return mean, std


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1, -2)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
