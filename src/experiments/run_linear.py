import torch
import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist
import pyro.contrib.autoguide as autoguide
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pyro.distributions.transforms import iterated, block_autoregressive, planar, radial, sylvester
from functools import partial
import tqdm
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
from src.experiments.setup import setup
device = setup()

results_path = "results/linear/"
model_path = "models/linear/"

problem = LinearRegression(n=2500, k=8, device=device)
data = problem.get_data()

model = problem.poisson_model
guide = normalizing_flow(model, flow_type=planar, num_flows=8)

train_result = train(
    data["train"],
    model,
    guide,
    epochs=1000,
    gradient_mc_samples=64,
    adam_params={"lr": 1e-2},
)



posterior_predictive = Predictive(model=model, guide=guide, num_samples=NUM_SAMPLES)
predictive_samples = posterior_predictive.get_samples(*data_ind)



data_dict = {k: np.expand_dims(v.squeeze().cpu().detach().numpy(), 0) for k, v in predictive_samples.items() if "obs" not in k and "y" not in k}
az_data = az.dict_to_dataset(data_dict)
summary = az.summary(az_data, round_to=4, kind="stats")

real_values = torch.cat((alpha_true.reshape(-1), beta_true, sigma_true.reshape(-1)))
summary["real"] = real_values
summary["in_bounds"] = (summary["real"] <= summary["hdi_97%"]) & (summary["real"] >= summary["hdi_3%"])

print(summary)



ax = sns.histplot(predictive_samples["sigma"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["sigma"], 0, 1000,c="r")
ax.set(title="Posterior samples $\\sigma$")

ax = sns.histplot(predictive_samples["alpha"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["alpha"], 0, 1000,c="r")
ax.set(title="Posterior samples $\\alpha$")



obs_data = az.dict_to_dataset({"obs": np.expand_dims(predictive_samples["obs"].squeeze().cpu().detach().numpy(), 0)})
obs_summary = az.summary(obs_data, round_to=4, kind="stats")

ax = sns.lineplot(x=age, y=obs_summary['mean'].values, linewidth=3, color="r")
ax.fill_between(age, obs_summary["hdi_97%"], obs_summary["hdi_3%"], color="r", alpha=0.5, label="94%")
sns.scatterplot(x=age, y=deaths, color="k", ax=ax)
ax.set(
    xlabel="Age",
    ylabel="Deaths",
    title="Posterior predictive distribution"
)

k_hat = psis_diagnostic(model, guide, *data)
print(k_hat)

k_hats = [psis_diagnostic(model, guide, *data) for _ in range(500)]
ax = sns.scatterplot(x=dist.Normal(0., .1).sample(torch.Size([len(k_hats)])), y=k_hats)
ax.axhline(0.7, ls='--', linewidth=3, color='r', label="0.7 threshold: poor fit")
ax.axhline(0.5, ls='-.', linewidth=3, color='r', alpha=0.5, label="0.5 threshold: possibly poor fit")
ax.set(
    title="Jittered scatterplot of $\hat{k}$",
    ylabel="$\hat{k}$"
)

plt.show()
