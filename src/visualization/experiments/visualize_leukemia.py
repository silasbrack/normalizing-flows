import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.setup import setup, save_plot
# setup()
from src.visualization.util import *

figure_path = "figures/leukemia/"

with open("results/leukemia/leukemia_mf_lr.pkl", "rb") as f:
    results = pickle.load(f)

print(results["samples"].keys())
print(results["samples"]["y"].shape)

# results["samples"].pop("beta")
# with open("results/leukemia/test.pkl", "wb") as f:
#     pickle.dump(results, f)

elbo = -results["losses"][-500:]
print("Final ELBO:", elbo[-500:].mean())

fig, ax = plt.subplots(figsize=(4, 4), ncols=1)
ax.plot(
    range(len(elbo)),
    elbo,
    color=PLANAR_COLOR,
)
ax.set(
    xlabel="Iteration",
    ylabel="ELBO",
    # xlim=[950, 1000],
    # ylim=[-10e17, -6e17],
)
finalize(ax)
fig.tight_layout()


fig, axs = plt.subplots(figsize=(12, 4), ncols=3)
axs[0].hist(
    results["samples"]["beta"][:, 1833].squeeze(),
    color=PLANAR_COLOR,
)
axs[0].set(
    xlabel="$\\beta_{1834}$",
    ylabel="Posterior density",
)
finalize(axs[0])
axs[1].hist(
    np.log(results["samples"]["lambda"][:, 1833].squeeze()),
    color=PLANAR_COLOR,
)
axs[1].set(
    xlabel="$\\log \\lambda_{1834}$",
)
finalize(axs[1])
axs[2].hist(
    np.log(results["samples"]["tau"].squeeze()),
    color=PLANAR_COLOR,
)
axs[2].set(
    xlabel="$\\log \\tau$",
)
finalize(axs[2])
adjust_spines(axs[0], ["left", "bottom"], adjust_loc=True)
adjust_spines(axs[1], ["bottom"], adjust_loc=True)
adjust_spines(axs[2], ["bottom"], adjust_loc=True)
fig.tight_layout()
# save_plot(figure_path, "")

plt.show()
