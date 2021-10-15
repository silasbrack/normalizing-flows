import matplotlib.pyplot as plt
from src.problems.energy_function import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

index = 0
for dist_name in ["U1", "U2", "U3", "U4"]:
    dist = EnergyPosteriorProblem.get_dist(dist_name)

    ax = plt.subplot(5, 5, index)
    plot_pdf(dist, ax=ax).set(
        title=dist_name,
    )
    index += 2

    for flow_type in [planar, radial]:
        for n_flows in [2, 8, 32]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(dist.dim) for _ in range(n_flows)],
            )
            train_result = problem.train(20000, adam_params={"lr": 5e-3})

            ax = plt.subplot(5, 9, index)
            plot_pdf(problem.log_prob, ax=ax).set(
                title=dist_name,
            )

            index += 1
        index += 1
