from src.visualization.setup import setup, save_plot
# setup()
from src.problems.poisson_regression import PoissonRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import arviz as az
from src.visualization.util import *


figure_path = "figures/poisson"


problem = PoissonRegression(device="cpu")
data = problem.get_data()

with open("results/poisson/poisson.pkl", "rb") as f:
    results = pickle.load(f)
predictive_samples = results["samples"]
losses = results["losses"]

fig, ax = plt.subplots(figsize=(4, 2.5))
ax.plot(
    range(len(losses)),
    -losses,
)
ax.set(
    xlabel="Iteration",
    ylabel="ELBO",
)
fig.tight_layout()
# plt.ylim([-500, 400])

# Create summary for latent variables
data_dict = {
    k: np.expand_dims(v, 0)
    for k, v in predictive_samples.items()
    if "obs" not in k and "y" not in k
}
az_data = az.dict_to_dataset(data_dict)
summary = az.summary(az_data, round_to=4, kind="stats")
print(summary)

# Create summary for observed variables
obs_data = az.dict_to_dataset({"obs": np.expand_dims(predictive_samples["obs"], 0)})
obs_summary = az.summary(obs_data, round_to=4, kind="stats")

fig, ax = plt.subplots(figsize=(3, 2.5))
ax.plot(data["age"], obs_summary['mean'].values, linewidth=3, color="r", label="Mean")
ax.fill_between(data["age"], obs_summary["hdi_97%"], obs_summary["hdi_3%"], color="r", alpha=0.33)
sns.scatterplot(x=data["age"], y=data["deaths"], color="k", ax=ax, label="Observations")
ax.set(
    xlabel="Age",
    ylabel="Deaths",
)
fig.tight_layout()
save_plot(figure_path, "posterior_predictive")

plt.show()
