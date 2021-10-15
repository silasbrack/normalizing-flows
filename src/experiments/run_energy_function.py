import matplotlib.pyplot as plt
from src.problems.energy_function import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

index = 1
for dist_name in ["U2", "U1", "U3", "U4"]:
    dist = EnergyPosteriorProblem.get_dist(dist_name)

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(5, 5, index)
    plot_pdf(dist, ax=ax).set(
        title=dist_name,
        xticks=[],
        yticks=[],
    )
    index += 2

    for flow_type in [planar, radial]:
        for n_flows in [2, 8, 32]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(dist.dim) for _ in range(n_flows)],
            )
            train_result = problem.train(1000, plot=False)

            ax = plt.subplot(5, 9, index)
            problem.plot_flow_samples(ax=ax)
            ax.set(
                title=f"K = {n_flows}",
                xticks=[],
                yticks=[],
            )

            index += 1
        index += 1
