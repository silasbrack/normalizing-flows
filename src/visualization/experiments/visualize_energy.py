from src.visualization.setup import setup
setup()
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme("paper", "ticks")
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize_distribution import plot_pdf, plot_samples
import tikzplotlib
import numpy as np
import pickle
import pandas as pd

final_elbos = {"Planar": [], "Radial": []}
idx = pd.MultiIndex.from_product([['Planar','Radial'], [2,8,32]])
df : pd.DataFrame = pd.DataFrame(np.random.randn(4,6), columns=idx, index=["U1", "U2", "U3", "U4"])

# plt.figure(figsize=(5,4))
for dist_name in ["U1", "U2", "U3", "U4"]:
    for name in ["Planar", "Radial"]:
        for n_flows in [2, 8, 32]:
            with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
                results = pickle.load(f)

            flow_samples = results["samples"]
            intermediate_samples = results["intermediate_samples"]
            losses = results["losses"]
            average_final_elbo = -np.mean(losses[-1000:])
            df[name][n_flows][dist_name] = average_final_elbo

            if dist_name == "U1":
                final_elbos[name].append(average_final_elbo)
                # plt.plot(-losses, label=f"{n_flows} {name.lower()} flows", linewidth=0.5)

df.index = ["$U_1(z)$", "$U_2(z)$", "$U_3(z)$", "$U_4(z)$"]
with open("figures/energy/energy_results_table.tex", "w") as f:
    df.to_latex(f, float_format="{:0.2f}".format)

# plt.ylabel("ELBO")
# plt.xlabel("Iteration")
# plt.ylim([-500, 400]) # U1
# # plt.ylim([-400, 800]) # U2
# # plt.ylim([-200, 1000]) # U3
# sns.despine()
# plt.legend()
# # plt.tight_layout()
# plt.savefig(f"figures/energy/energy_training_curves_{dist_name}.pgf",
#             backend="pgf",
#             dpi=1000,
#             bbox_inches='tight',
# )
#

df = df.reset_index().rename(columns={"index": "type"}) \
       .melt(id_vars="type", value_name="ELBO", var_name=["flow", "n_flows"]) \
       .assign(idx=lambda df: df["n_flows"].map({2: 0, 8: 1, 32: 2}))
g = sns.FacetGrid(df, col="type",  col_wrap=2, sharey=False, height=2.5, aspect=3/2.5)
g.map(sns.scatterplot, "idx", "ELBO", "flow", palette=["r", "b"])
g.map(sns.lineplot, "idx", "ELBO", "flow", palette=["r", "b"], linewidth=2, linestyle="--")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.add_legend()
g.set_axis_labels("Number of flows", "ELBO")
plt.xticks([0,1,2], ["2","8","32"])
sns.despine()
plt.savefig(f"figures/energy/final_elbo_comparison.pgf",
            backend="pgf",
            dpi=1000,
            bbox_inches='tight',
)
tikzplotlib.save(f"figures/energy/final_elbo_comparison.tex")

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



"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

# plt.figure(figsize=(20, 10))
plt.figure(figsize=(10, 5))
# plt.figure(figsize=(5, 2.5))

index = 1
for dist_name in ["U1", "U2", "U3", "U4"]:
    # pyro.get_param_store().load(f"models/energy/{}.save")

    dist = EnergyPosteriorProblem.get_dist(dist_name)

    # ax = plt.subplot(4, 9, index)
    ax = plt.subplot(5, 7, index)
    plot_pdf(dist, ax=ax, how="contour").set(
        # title=dist_name,
        ylabel=dist_name,
        xticks=[],
        yticks=[],
    )
    # index += 2
    index += 1

    for flow_type, name in [(planar, "planar"), (radial, "radial")]:
        for n_flows in [2, 8, 32]:
            with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
                results = pickle.load(f)
            flow_samples = results["samples"]

            # ax = plt.subplot(4, 9, index)
            ax = plt.subplot(5, 7, index)
            plot_samples(samples=flow_samples, ax=ax, shade=False)
            ax.set(
                title=f"K = {n_flows}" if dist_name == "U1" else None,
                xticks=[],
                yticks=[],
            )

            index += 1
    #     index += 1
    # index -= 1
# plt.tight_layout()
# tikzplotlib.clean_figure()
# tikzplotlib.save("figures/energy/energy_grid.tex")
plt.savefig("figures/energy/energy_grid.pgf", backend="pgf")
plt.savefig("figures/energy/energy_grid.pdf", backend="pgf")

plt.show()
