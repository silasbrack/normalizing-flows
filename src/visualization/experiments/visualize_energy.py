import matplotlib.pyplot as plt
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf, plot_samples
from src.visualization.setup import setup
import pickle
setup()

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

index = 1
for dist_name in ["U1", "U2", "U3", "U4"]:
    dist = EnergyPosteriorProblem.get_dist(dist_name)

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(5, 9, index)
    plot_pdf(dist, ax=ax).set(
        title=dist_name,
        xticks=[],
        yticks=[],
    )
    index += 2

    results = pickle.load()

    for flow_type in [planar, radial]:
        for n_flows in [2, 8, 32]:
            flow_samples = results["flow_samples"][flow_type][n_flows]

            ax = plt.subplot(5, 9, index)
            plot_samples(flow_samples, ax=ax)
            ax.set(
                title=f"K = {n_flows}",
                xticks=[],
                yticks=[],
            )

            index += 1
        index += 1
    index -= 1
plt.tight_layout()
plt.show()
