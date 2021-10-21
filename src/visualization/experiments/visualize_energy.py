import matplotlib.pyplot as plt
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf, plot_samples
from src.visualization.setup import setup
import tikzplotlib
import numpy as np
setup()

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

plt.figure(figsize=(20, 10))

index = 1
for dist_name in ["U1", "U2", "U3"]:
    dist = EnergyPosteriorProblem.get_dist(dist_name)

    ax = plt.subplot(5, 9, index)
    # ax = plt.subplot(5, 7, index)
    plot_pdf(dist, ax=ax, how="contour").set(
        title=dist_name,
        xticks=[],
        yticks=[],
    )
    index += 2
    # index += 1

    for flow_type, name in [(planar, "planar"), (radial, "radial")]:
        for n_flows in [2, 8, 32]:
            results = np.load("results/energy/" + f"{dist_name}_{name}_{n_flows}.npy")
            #flow_samples = results["flow_samples"][flow_type][n_flows]
            flow_samples = results

            ax = plt.subplot(5, 9, index)
            # ax = plt.subplot(5, 7, index)
            plot_samples(samples=flow_samples, ax=ax, shade=False)
            ax.set(
                title=f"K = {n_flows}" if dist_name == "U1" else None,
                xticks=[],
                yticks=[],
            )

            index += 1
        index += 1
    index -= 1
plt.tight_layout()
# tikzplotlib.save("figures/energy/energy_grid.tex")
plt.savefig("figures/energy/energy_grid.pdf") # backend="pgf"
plt.show()
