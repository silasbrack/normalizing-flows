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
    epochs=10000,
    gradient_mc_samples=64,
    adam_params={"lr": 1e-2},
)
plt.show()

# Sample from posterior
posterior_predictive = Predictive(model=model, guide=guide, num_samples=1024)
predictive_samples = posterior_predictive.get_samples(*data["test"])

# Create summary for latent variables
data_dict = {
    k: np.expand_dims(v.squeeze().cpu().detach().numpy(), 0)
    for k, v in predictive_samples.items()
    if "obs" not in k and "y" not in k
}
az_data = az.dict_to_dataset(data_dict)
summary = az.summary(az_data, round_to=4, kind="stats")

real_values = torch.cat((problem.true_params["alpha"].reshape(-1),
                         problem.true_params["beta"],
                         problem.true_params["sigma"].reshape(-1)))
summary["real"] = real_values
summary["in_bounds"] = (summary["real"] <= summary["hdi_97%"]) & (summary["real"] >= summary["hdi_3%"])

print(summary)

# Plot latent variable histograms
ax = sns.histplot(predictive_samples["sigma"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["sigma"], 0, 1000, c="r")
ax.set(title="Posterior samples $\\sigma$")
plt.show()

ax = sns.histplot(predictive_samples["alpha"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["alpha"], 0, 1000, c="r")
ax.set(title="Posterior samples $\\alpha$")
plt.show()

# Calculate k hat statistic
k_hat = psis_diagnostic(model, guide, *data["train"])
if k_hat < 0.5:
    print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
elif k_hat < 0.7:
    print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
else:
    print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")

# k_hats = [psis_diagnostic(model, guide, *data["train"]) for _ in range(500)]
# ax = sns.scatterplot(x=dist.Normal(0., .1).sample(torch.Size([len(k_hats)])), y=k_hats)
# ax.axhline(0.7, ls='--', linewidth=3, color='r', label="0.7 threshold: poor fit")
# ax.axhline(0.5, ls='-.', linewidth=3, color='r', alpha=0.5, label="0.5 threshold: possibly poor fit")
# ax.set(
#     title="Jittered scatterplot of $\\hat{k}$",
#     ylabel="$\\hat{k}$"
# )
# plt.show()
