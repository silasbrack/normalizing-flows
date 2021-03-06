from src.problems.energy import EnergyPosteriorProblem
import pyro
import torch
from pyro.distributions.transforms import planar, radial
from src.experiments.setup import setup
import pickle
device = setup()


"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

results_path = "results/energy/"
model_path = "models/energy/"

epochs = 10000
n_samples = 1024
for dist_name in ["U3"]:
    for flow_type, name in [(radial, "radial")]:
        for n_flows in [32]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(input_dim=2) for _ in range(n_flows)],
                device=device,
            )
            train_result = problem.train(
                epochs,
                plot=True,
                gradient_mc_samples=16,
                adam_params={"lr": 5e-3},
            )
            samples = problem.sample(n_samples)
            log_prob = problem.log_prob(samples)

            train_result["samples"] = {
                "samples": samples.cpu().detach().numpy(),
                "log_prob": log_prob.cpu().detach().numpy(),
            }

            file_name = f"{dist_name}_{name}_{n_flows}"
            with open(results_path + file_name + ".pkl", "wb") as f:
                pickle.dump(train_result, f)
            pyro.get_param_store().save(model_path + file_name + ".save")
