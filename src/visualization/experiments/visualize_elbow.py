from src.visualization.setup import setup, save_plot
setup()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd

figure_path = "figures/energy"

dist_name = "U1"
name = "Planar"
n_flows = 32

with open("results/energy/" + f"{dist_name}_{name.lower()}_{n_flows}.pkl", "rb") as f:
    results = pickle.load(f)

fig, axes = plt.subplots(1, 3, figsize=(7.5,2.5))

ax = sns.lineplot(x=range(len(results["losses"])), y=-results["losses"], color="k", ax=axes[0])
ax.set(title="Training curve", xlabel="Iterations", ylabel="ELBO", xlim=[4500, 6000], ylim=[-50,350])

intermediate_samples = results["intermediate_samples"][5000]
data = pd.DataFrame({"x": intermediate_samples["samples"][:, 0], "y": intermediate_samples["samples"][:, 1], "log_prob": intermediate_samples["log_prob"]})
# ax = sns.scatterplot(data=data, x="x", y="y", color="k", s=15, ax=axes[1])
ax = sns.scatterplot(data=data, x="x", y="y", hue="log_prob", palette="rocket", s=15, ax=axes[1])
ax.set(title="Iteration 5000", xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlim=[-4,4], ylim=[-4,4])
ax.get_legend().remove()

intermediate_samples = results["intermediate_samples"][6000]
data = pd.DataFrame({"x": intermediate_samples["samples"][:, 0], "y": intermediate_samples["samples"][:, 1], "log_prob": intermediate_samples["log_prob"]})
# ax = sns.scatterplot(data=data, x="x", y="y", color="k", s=15, ax=axes[2])
ax = sns.scatterplot(data=data, x="x", y="y", hue="log_prob", palette="rocket", s=15, ax=axes[2])
ax.set(title="Iteration 6000", xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlim=[-4,4], ylim=[-4,4])
ax.get_legend().remove()

# plt.tight_layout()
plt.subplots_adjust(bottom=0.20, left=0.10, right=0.90)

save_plot(figure_path, "elbow")
plt.show()
