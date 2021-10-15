import torch
import pyro
from pyro import distributions as dist


class MultivariateNormal:
    def __init__(self, n=5000):
        self.n = n

    def model(self, data=None):
        loc_of_loc = torch.zeros(2)
        scale_of_loc = torch.ones(2)

        # Priors
        loc = pyro.sample("loc", dist.MultivariateNormal(loc_of_loc, torch.diag(scale_of_loc)))
        scale = torch.eye(2, 2)
        # scale = torch.diag(pyro.sample("scale", dist.LogNormal(torch.zeros(2), torch.ones(2)).to_event(1)))

        # Posterior model
        with pyro.plate("data", self.n):
            return pyro.sample("x", dist.MultivariateNormal(loc, scale), obs=data)

    def get_data(self, loc=torch.Tensor([2., -6.]), scale=torch.Tensor([[2.9, -2.6], [-2.6, 3.8]])):
        true_distribution = dist.MultivariateNormal(loc, scale)
        x = true_distribution(torch.Size([self.n]))
        return {"data": x}
