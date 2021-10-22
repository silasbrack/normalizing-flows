from src.problems.energy import EnergyPosteriorProblem
import pyro
import torch
from pyro.distributions.transforms import planar, radial
from src.experiments.setup import set_seeds
import pickle
set_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

results_path = "results/energy/"
model_path = "models/energy/"

epochs = 20000
n_samples = 256
for dist_name in ["U1"]:
    for flow_type, name in [(planar, "planar")]:
        for n_flows in [16]:
            problem = EnergyPosteriorProblem(
                dist_name=dist_name,
                nfs=[flow_type(input_dim=2) for _ in range(n_flows)],
                device=device,
            )
            train_result = problem.train(
                epochs,
                plot=False,
                gradient_mc_samples=16,
                adam_params={"lr": 5e-3},
            )
            samples = problem.sample(n_samples).cpu().detach().numpy()

            train_result["samples"] = samples

            file_name = f"{dist_name}_{name}_{n_flows}"
            with open(results_path + file_name + ".pkl", "wb") as f:
                pickle.dump(train_result, f)
            pyro.get_param_store().save(model_path + file_name + ".save")
