import torch
from pyro import distributions as dist

def model(data=None, n=1):
    loc_of_loc = torch.zeros(2)
    scale_of_loc = torch.ones(2)

    # Priors
    loc = pyro.sample("loc", dist.MultivariateNormal(loc_of_loc, torch.diag(scale_of_loc)))
    scale = torch.diag(pyro.sample("scale", dist.LogNormal(torch.zeros(2), torch.ones(2)).to_event(1)))

    # Posterior model
    with pyro.plate("data", n):
        return pyro.sample("x", dist.MultivariateNormal(loc, scale), obs=data)

def get_data(n = 5000, loc = torch.Tensor([2., -6.]), scale = torch.Tensor([[2.9, -2.6], [-2.6, 3.8]])):
    true_distribution = dist.MultivariateNormal(loc, scale)
    x = true_distribution(torch.Size([n]))
    return x
