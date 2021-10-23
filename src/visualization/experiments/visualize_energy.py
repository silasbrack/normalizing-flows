from src.visualization.setup import setup
setup()
import matplotlib.pyplot as plt
import seaborn as sns
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf, plot_samples
import tikzplotlib
import numpy as np
import pickle

final_elbos = {"planar": [], "radial": []}

plt.figure(figsize=(5,4))
dist_name = "U1"
for name in ["planar", "radial"]:
    for n_flows in [2, 8, 32]:
        with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
            results = pickle.load(f)

        flow_samples = results["samples"]
        intermediate_samples = results["intermediate_samples"]
        losses = results["losses"]
        average_final_elbo = -np.mean(losses[-1000:])
        final_elbos[name].append(average_final_elbo)

        plt.plot(-losses, label=f"{n_flows} {name} flows", linewidth=0.5)

plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.ylim([-500, 400]) # U1
# plt.ylim([-400, 800]) # U2
# plt.ylim([-200, 1000]) # U3
sns.despine()
plt.legend()
# plt.tight_layout()
plt.savefig(f"figures/energy/energy_training_curves_{dist_name}.pgf",
            backend="pgf",
            dpi=1000,
            bbox_inches='tight',
)

plt.figure(figsize=(3,2.5))
ax = sns.scatterplot(x=[0,1,2], y=final_elbos["planar"], color="r", s=30, label="Planar")
sns.lineplot(x=[0,1,2], y=final_elbos["planar"], color="r", linewidth=2, linestyle="--")
sns.scatterplot(x=[0,1,2], y=final_elbos["radial"], color="b", s=30, label="Radial")
sns.lineplot(x=[0,1,2], y=final_elbos["radial"], color="b", linewidth=2, linestyle="--")
plt.ylabel("ELBO")
plt.xlabel("Number of flows")
plt.xticks([0,1,2], ["2","8","32"])
sns.despine()
plt.legend()
# plt.tight_layout()
plt.savefig(f"figures/energy/final_elbo_comparison_{dist_name}.pgf",
            backend="pgf",
            dpi=1000,
            bbox_inches='tight',
)
tikzplotlib.save(f"figures/energy/final_elbo_comparison_{dist_name}.tex")

fig, ax = plt.subplots(figsize=(3,2.5))
dist_name = "U1"
name = "planar"
n_flows = 32
with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
    results = pickle.load(f)
flow_samples = results["samples"]
dist = EnergyPosteriorProblem.get_dist(dist_name)
ax = sns.scatterplot(
    x=flow_samples[:, 0],
    y=flow_samples[:, 1],
    color="k",
)
plot_pdf(dist, ax=ax, how="contour").set(
    title=dist_name,
    xticks=[],
    yticks=[],
)
plt.savefig(f"figures/energy/{dist_name}.pgf",
            backend="pgf",
            dpi=1000,
            bbox_inches='tight',
)
tikzplotlib.save(f"figures/energy/{dist_name}.tex")


# """
# We want to recreate figure 3 from Rezende et al, 2015.
# """
#
# # plt.figure(figsize=(20, 10))
# plt.figure(figsize=(10, 5))
# # plt.figure(figsize=(5, 2.5))
#
# index = 1
# for dist_name in ["U1", "U2", "U3", "U4"]:
#     # pyro.get_param_store().load(f"models/energy/{}.save")
#
#     dist = EnergyPosteriorProblem.get_dist(dist_name)
#
#     # ax = plt.subplot(4, 9, index)
#     ax = plt.subplot(5, 7, index)
#     plot_pdf(dist, ax=ax, how="contour").set(
#         # title=dist_name,
#         ylabel=dist_name,
#         xticks=[],
#         yticks=[],
#     )
#     # index += 2
#     index += 1
#
#     for flow_type, name in [(planar, "planar"), (radial, "radial")]:
#         for n_flows in [2, 8, 32]:
#             results = np.load("results/energy/" + f"{dist_name}_{name}_{n_flows}.npy")
#             flow_samples = results
#             # with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
#             #     results = pickle.load(f)
#             # flow_samples = results["samples"]
#
#             # ax = plt.subplot(4, 9, index)
#             ax = plt.subplot(5, 7, index)
#             plot_samples(samples=flow_samples, ax=ax, shade=False)
#             ax.set(
#                 title=f"K = {n_flows}" if dist_name == "U1" else None,
#                 xticks=[],
#                 yticks=[],
#             )
#
#             index += 1
#     #     index += 1
#     # index -= 1
# # plt.tight_layout()
# # tikzplotlib.clean_figure()
# # tikzplotlib.save("figures/energy/energy_grid.tex")
# plt.savefig("figures/energy/energy_grid.pgf", backend="pgf") # backend="pgf"
# plt.show()

plt.show()
