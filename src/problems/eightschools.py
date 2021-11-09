import torch
import pyro
from pyro import distributions as dist


class EightSchools:
    def __init__(self, device):
        self.device = device
        self.N = torch.tensor(8).to(self.device)
        self.J = torch.tensor(8).to(self.device)

    def model(self, sigma, y=None):
        mu = pyro.sample('mu', dist.Normal(torch.tensor(0.).to(self.device), torch.tensor(5.).to(self.device))).to(self.device)
        tau = pyro.sample('tau', dist.HalfCauchy(torch.tensor(5.).to(self.device))).to(self.device)
        with pyro.plate('J', self.J):
            theta = pyro.sample('theta', dist.Normal(mu, tau)).to(self.device)
            pyro.sample('obs', dist.Normal(theta, sigma), obs=y).to(self.device)

    def get_data(self):
        y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]).to(self.device)
        sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]).to(self.device)

        return {"y": y, "sigma": sigma, "train": (sigma, y), "test": (sigma,)}
