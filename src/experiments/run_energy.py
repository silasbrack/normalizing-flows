from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

epochs = 20000
flow_samples = 256
for dist_name in ["U1", "U2", "U3", "U4"]:
    for flow_type in [planar, radial]:
        for n_flows in [2, 8, 32]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(input_dim=2) for _ in range(n_flows)],
            )
            train_result = problem.train(epochs, plot=False)
            flow_samples = problem.sample((flow_samples,))
