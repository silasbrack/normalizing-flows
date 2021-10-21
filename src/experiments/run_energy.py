from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
import numpy as np

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

results_folder = "results/"
experiment_folder = "energy/"
path = results_folder + experiment_folder

epochs = 20000
n_samples = 256
for dist_name in ["U1", "U2", "U3", "U4"]:
    for flow_type, name in [(planar, "planar"), (radial, "radial")]:
        for n_flows in [2, 8, 32]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(input_dim=2) for _ in range(n_flows)],
            )
            train_result = problem.train(epochs, plot=False)
            samples = problem.sample(n_samples)
            np.save(path + f"{dist_name}_{name}_{n_flows}.npy", samples.cpu().detach().numpy())