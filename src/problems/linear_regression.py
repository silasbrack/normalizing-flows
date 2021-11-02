import torch
import pyro
from pyro import distributions as dist


class LinearRegression:
    def __init__(self, n, k, device):
        self.N = n
        self.dim = k
        self.true_params = None
        self.device = device

    def model(self, x, y=None):
        alpha = pyro.sample("alpha", dist.Normal(0.0, 100.0)).to(self.device)
        with pyro.plate("betas", self.dim):
            beta = pyro.sample("beta", dist.Normal(0, 10.0)).unsqueeze(0).to(self.device)
        sigma = pyro.sample("sigma", dist.HalfNormal(10.0)).to(self.device)
        with pyro.plate("yn"):
            mu = beta @ x.T + alpha
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    def get_data(self):
        alpha_true = dist.Normal(42.0, 10.0).sample()
        beta_true = dist.Normal(torch.zeros(self.dim), 10.0).sample()
        sigma_true = dist.Exponential(1.0).sample()
        self.true_params = {"alpha": alpha_true, "beta": beta_true, "sigma": sigma_true}

        eps = dist.Normal(0.0, sigma_true).sample([self.N])
        x = torch.randn(self.N, self.dim)
        y = alpha_true + x @ beta_true + eps

        x = x.to(self.device)
        y = y.unsqueeze(0).to(self.device)

        return {"X": x, "y": y, "train": (x, y), "test": (x,)}
