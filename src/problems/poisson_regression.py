import torch
import pyro
from pyro import distributions as dist
from sklearn.preprocessing import scale


class PoissonRegression:
    def __init__(self, device):
        self.device = device

    def model(self, x, y=None):
        kappa2 = pyro.sample("kappa2", dist.LogNormal(0, 1))
        w0 = pyro.sample("w0", dist.Normal(0, kappa2)).unsqueeze(0).to(self.device)
        w1 = pyro.sample("w1", dist.Normal(0, kappa2)).unsqueeze(0).to(self.device)
        with pyro.plate("yn"):
            obs = pyro.sample("obs", dist.Poisson(torch.exp(w1 @ x.T + w0)), obs=y)

    def get_data(self):
        age = torch.arange(35, 65)
        deaths = torch.tensor(
            [3, 1, 3, 2, 2, 4, 4, 7, 5, 2, 8, 13, 8, 2, 7, 4, 7, 4, 4, 11, 11, 13, 12, 12, 19, 12, 16, 12, 6, 10])

        x = torch.tensor(scale(age)).unsqueeze(1).to(self.device)
        y = deaths.unsqueeze(0).to(self.device)

        return {"age": age, "deaths": deaths, "X": x, "y": y, "train": (x, y), "test": (x,)}
