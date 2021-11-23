import torch
import pyro
from pyro import distributions as dist
import pandas as pd
import numpy as np


class Leukemia:
    def __init__(self, n=72, j=7128, tau0=None, device="cpu"):
        self.device = device
        self.N = torch.tensor(n).to(self.device)
        self.J = torch.tensor(j).to(self.device)

        if tau0 is None:
            tau0 = 1 / (self.J - 1) * 2 / self.N.sqrt()

        self.tau0 = tau0

    def model(self, x, y=None):
        c2 = pyro.sample("c2", dist.InverseGamma(2, 8))
        tau = pyro.sample("tau", dist.HalfCauchy(self.tau0))
        tau2 = tau**2

        with pyro.plate("J", self.J):
            lambda_ = pyro.sample("lambda", dist.HalfCauchy(1))
            lambda_tilde2 = c2*lambda_**2 / (c2 + tau2 * lambda_**2)
            beta = pyro.sample("beta", dist.Normal(0, tau2 * lambda_tilde2))

        logits = beta @ x.T

        with pyro.plate("N", self.N):
            pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    def get_data(self):
        df = pd.read_csv("data/leukemia.csv")
        df = df.T.reset_index()
        df[["type", "index"]] = df["index"].str.split(".", 1, expand=True).fillna(0)

        y = df["type"].to_numpy()
        _, y = np.unique(y, return_inverse=True)
        x = df[list(range(7128))].to_numpy()

        y = torch.tensor(y).float()
        x = torch.tensor(x)

        return {"y": y, "X": x, "train": (x, y), "test": (x,)}
