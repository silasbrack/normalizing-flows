import torch
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pyro.distributions.transforms import iterated, block_autoregressive, planar, radial, sylvester
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
import pickle
from src.problems import LinearRegression
from src.guides import normalizing_flow
from src.experiments import train, setup
device = setup()

results_path = "results/linear/"
model_path = "models/linear/"

problem = LinearRegression(n=2500, k=8, device=device)
data = problem.get_data()

model = problem.model
guide = normalizing_flow(model, flow_type=planar, num_flows=4)

train_result = train(
    data["train"],
    model,
    guide,
    epochs=30000,
    gradient_mc_samples=512,
    adam_params={"lr": 1e-2},
)

# Sample from posterior
posterior_predictive = Predictive(model=model, guide=guide, num_samples=1024)
predictive_samples = posterior_predictive(*data["test"])
predictive_samples = {k: v.cpu().detach().numpy() for k, v in predictive_samples.items()}
train_result["samples"] = predictive_samples

# # Calculate k hat statistic
# k_hat = psis_diagnostic(model, guide, *data["train"])
# if k_hat < 0.5:
#     print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
# elif k_hat < 0.7:
#     print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
# else:
#     print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")

file_name = "linear"
with open(results_path + file_name + ".pkl", "wb") as f:
    pickle.dump(train_result, f)
pyro.get_param_store().save(model_path + file_name + ".save")
