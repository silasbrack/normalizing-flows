import pickle
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.setup import setup, save_plot
# setup()
from src.visualization.util import *


figure_path = "figures/energy"

dist_name = "U1"
name = "Planar"
n_flows = 32

with open("results/energy/" + f"{dist_name}_{name.lower()}_{n_flows}.pkl", "rb") as f:
    results = pickle.load(f)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

axes[0].plot(range(len(results["losses"])), -results["losses"], color="k")
axes[0].set(title="Training curve", xlabel="Iterations", ylabel="ELBO", xlim=[4500, 6000], ylim=[-50, 350])

intermediate_samples = results["intermediate_samples"][5000]
data = pd.DataFrame({
    "x": intermediate_samples["samples"][:, 0],
    "y": intermediate_samples["samples"][:, 1],
    "log_prob": intermediate_samples["log_prob"],
})
axes[1].scatter(data["x"], data["y"], c=data["log_prob"], cmap="rocket", s=15)
axes[1].set(
    title="Iteration 6000",
    xlabel=None,
    ylabel=None,
    xticks=[],
    yticks=[],
    xticklabels=[],
    yticklabels=[],
    xlim=[-4, 4],
    ylim=[-4, 4],
)

intermediate_samples = results["intermediate_samples"][6000]
data = pd.DataFrame({
    "x": intermediate_samples["samples"][:, 0],
    "y": intermediate_samples["samples"][:, 1],
    "log_prob": intermediate_samples["log_prob"],
})
axes[2].scatter(data["x"], data["y"], c=data["log_prob"], cmap="rocket", s=15)
axes[2].set(
    title="Iteration 6000",
    xlabel=None,
    ylabel=None,
    xticks=[],
    yticks=[],
    xticklabels=[],
    yticklabels=[],
    xlim=[-4, 4],
    ylim=[-4, 4],
)

# plt.tight_layout()
plt.subplots_adjust(bottom=0.20, left=0.10, right=0.90)

save_plot(figure_path, "elbow")
plt.show()
