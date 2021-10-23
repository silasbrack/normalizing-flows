import seaborn as sns
import matplotlib.pyplot as plt
import pyro
from pyro.distributions.transforms import planar
from src.experiments.train import train
from src.guides import normalizing_flow
from src.problems.poisson_regression import PoissonRegression
from src.experiments.setup import set_seeds
from pyro.infer import Predictive
import pickle
set_seeds()

pyro.get_param_store().load("models/poisson/poisson.save")

print(pyro.get_param_store().named_parameters())

problem = PoissonRegression()
data = problem.get_data()

model = problem.poisson_model
guide = normalizing_flow(model, flow_type=planar, num_flows=4)

n_samples = 4096
posterior_predictive = Predictive(model, guide=guide, num_samples=n_samples)
predictive_samples = posterior_predictive.get_samples(data["design_matrix"]) # Just remember that this breaks the log_prob of the posterior samples
prediction = predictive_samples["y"].squeeze()#.reshape(torch.Size([N*n_samples, 2]))

