import matplotlib.pyplot as plt
import seaborn as sns
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
from src.visualization.visualize import plot_pdf
import numpy as np
import pickle
import pandas as pd
import tikzplotlib
from src.visualization.setup import setup, save_plot
setup()


figure_path = "figures/energy/"

final_elbos = {"Planar": [], "Radial": []}
idx = pd.MultiIndex.from_product([['Planar', 'Radial'], [2, 8, 32]])
df = pd.DataFrame(np.random.randn(4, 6), columns=idx, index=["U1", "U2", "U3", "U4"])

for dist_name in ["U1", "U2", "U3", "U4"]:
    for name in ["Planar", "Radial"]:
        for n_flows in [2, 8, 32]:
            with open("results/energy/" + f"{dist_name}_{name.lower()}_{n_flows}.pkl", "rb") as f:
                results = pickle.load(f)

            losses = results["losses"]
            average_final_elbo = -np.mean(losses[-1000:])
            df.loc[dist_name][name][n_flows] = average_final_elbo

            if dist_name == "U1":
                final_elbos[name].append(average_final_elbo)

df.index = ["$U_1(z)$", "$U_2(z)$", "$U_3(z)$", "$U_4(z)$"]
with open("figures/energy/energy_results_table.tex", "w") as f:
    df.to_latex(f, float_format="{:0.2f}".format)

fig, axs = plt.subplots(nrows=2, ncols=2)
i = 0
for row in axs:
    for col in row:
        planar_vals = df.loc["$U_" + str(i+1) + "(z)$", ("Planar")]
        radial_vals = df.loc["$U_" + str(i+1) + "(z)$", ("Radial")]
        col.scatter(range(3), planar_vals.values)
        col.plot(range(3), planar_vals.values, linestyle="--", linewidth=2)
        col.scatter(range(3), radial_vals.values)
        col.plot(range(3), radial_vals.values, linestyle="--", linewidth=2)
        col.set_xticks(range(3))
        col.set_xticklabels(planar_vals.index.values)
        col.set_title(df.index.values[i])
        col.set_xlabel("Number of flows")
        col.set_ylabel("ELBO")
        i = i + 1
axs[0, 0].annotate(
    text="Planar",
    xy=(0.1, 190),
)
axs[0, 0].annotate(
    text="Radial",
    xy=(0.5, 50),
)

plt.tight_layout()
tikzplotlib.clean_figure()
tikzplotlib.save(figure_path + "final_elbo_comparison" + ".tex", axis_height="\\figureheight", axis_width="\\figurewidth")



"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

plt.figure(figsize=(8, 4))
index = 1
for dist_name in ["U1", "U2", "U3", "U4"]:
    # pyro.get_param_store().load(f"models/energy/{}.save")

    dist = EnergyPosteriorProblem.get_dist(dist_name)

    # ax = plt.subplot(4, 9, index)
    ax = plt.subplot(4, 7, index)
    plot_pdf(dist, ax=ax, how="contour").set(
        # title=dist_name,
        xlim=[-4, 4],
        ylim=[-4, 4],
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
            log_prob = results["samples"]["log_prob"]

            # ax = plt.subplot(4, 9, index)
            ax = plt.subplot(4, 7, index)
            # plot_samples(samples=flow_samples, ax=ax, shade=False)
            ax.scatter(
                x=flow_samples[:, 0],
                y=flow_samples[:, 1],
                c=log_prob, cmap="rocket",
                s=3,
            )
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

plt.tight_layout()
save_plot(figure_path, "energy_grid")

plt.show()
