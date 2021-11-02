from src.visualization.setup import setup, save_plot
setup()
import matplotlib.pyplot as plt
import seaborn as sns
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize import plot_pdf
import numpy as np
import pickle
import pandas as pd

figure_path = "figures/energy"

final_elbos = {"Planar": [], "Radial": []}
idx = pd.MultiIndex.from_product([['Planar', 'Radial'], [2,8,32]])
df = pd.DataFrame(np.random.randn(4, 6), columns=idx, index=["U1", "U2", "U3", "U4"])

for dist_name in ["U1", "U2", "U3", "U4"]:
    for name in ["Planar", "Radial"]:
        for n_flows in [2, 8, 32]:
            with open("results/energy/" + f"{dist_name}_{name.lower()}_{n_flows}.pkl", "rb") as f:
                results = pickle.load(f)

            intermediate_samples = results["intermediate_samples"]
            losses = results["losses"]
            average_final_elbo = -np.mean(losses[-1000:])
            df.loc[dist_name][name][n_flows] = average_final_elbo

            if dist_name == "U1":
                final_elbos[name].append(average_final_elbo)

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
# save_plot(figure_path, "final_elbo_comparison", ["pgf", "pdf"])
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
save_plot(figure_path, "final_elbo_comparison")

fig, ax = plt.subplots(figsize=(3,2.5))
dist_name = "U1"
name = "planar"
n_flows = 32
with open("results/energy/" + f"{dist_name}_{name}_{n_flows}.pkl", "rb") as f:
    results = pickle.load(f)
flow_samples = results["samples"]["samples"]
dist = EnergyPosteriorProblem.get_dist(dist_name)
ax = sns.scatterplot(
    x=flow_samples[:, 0],
    y=flow_samples[:, 1],
    color="k",
    ax=ax,
)
plot_pdf(dist, ax=ax, how="contour").set(
    title=dist_name,
    xticks=[],
    yticks=[],
    )
save_plot(figure_path, dist_name)


"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

plt.figure(figsize=(10, 5))
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
            flow_samples = results["samples"]["samples"]

            # ax = plt.subplot(4, 9, index)
            ax = plt.subplot(5, 7, index)
            # plot_samples(samples=flow_samples, ax=ax, shade=False)
            sns.scatterplot(x=flow_samples[:, 0], y=flow_samples[:, 1], cmap=sns.color_palette("rocket", as_cmap=True), ax=ax)
            ax.set(
                title=f"K = {n_flows}" if dist_name == "U1" else None,
                xlim=[-4, 4],
                ylim=[-4, 4],
                xticks=[],
                yticks=[],
            )

            index += 1
    #     index += 1
    # index -= 1
# plt.tight_layout()


plt.show()
