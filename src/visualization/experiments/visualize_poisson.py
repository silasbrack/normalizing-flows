from src.visualization.setup import setup
setup()
from src.problems.poisson_regression import PoissonRegression
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib
import pickle
import numpy as np
import arviz as az


problem = PoissonRegression(device="cpu")
data = problem.get_data()

with open("results/poisson/poisson.pkl", "rb") as f:
    results = pickle.load(f)
predictive_samples = results["samples"]
losses = results["losses"]

plt.figure(figsize=(4, 2.5))
sns.lineplot(
    x=range(len(losses)),
    y=-losses,
).set(
    xlabel="Iteration",
    ylabel="ELBO",
)
# plt.ylim([-500, 400])
sns.despine()


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

plt.figure(figsize=(3, 2.5))
ax = sns.lineplot(x=data["age"], y=obs_summary['mean'].values, linewidth=3, color="r", label="Mean")
ax.fill_between(data["age"], obs_summary["hdi_97%"], obs_summary["hdi_3%"], color="r", alpha=0.5, label="94% HDI")
sns.scatterplot(x=data["age"], y=data["deaths"], color="k", ax=ax, label="Observations")
ax.set(
    xlabel="Age",
    ylabel="Deaths",
)
sns.despine()
plt.savefig("figures/poisson/posterior_predictive.pgf", backend="pgf")
plt.savefig("figures/poisson/posterior_predictive.pdf", backend="pgf")
tikzplotlib.save("figures/poisson/posterior_predictive.tex")

plt.show()
