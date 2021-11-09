import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyro
from pyro.distributions.transforms import planar
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
from src.guides import normalizing_flow
from src.problems import EightSchools
from src.experiments import setup
import arviz as az
device = setup()

pyro.clear_param_store()
pyro.get_param_store().load("models/eightschools/eightschools.save")

problem = EightSchools(device)
data = problem.get_data()

model = problem.model
guide = normalizing_flow(model, flow_type=planar, num_flows=16)

n_samples = 1024
posterior = Predictive(model, guide=guide, num_samples=n_samples)
posterior_samples = posterior(*data["test"])

k_hats = [psis_diagnostic(model, guide, *data["train"]) for _ in range(500)]
k_hat = np.mean(k_hats)

if k_hat < 0:
    print(f"k hat = {k_hat:.3f} < 0: Wtf?")
elif k_hat < 0.5:
    print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
elif k_hat < 0.7:
    print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
else:
    print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")



