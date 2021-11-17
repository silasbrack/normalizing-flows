import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.setup import setup, save_plot
# setup()
from src.visualization.util import *

figure_path = "figures/eightschools/"

# pyro.clear_param_store()
# pyro.get_param_store().load("models/eightschools/eightschools.save")

# problem = EightSchools(device)
# data = problem.get_data()

# model = problem.model
# guide = normalizing_flow(model, flow_type=planar, num_flows=16)
#
# n_samples = 1024
# posterior = Predictive(model, guide=guide, num_samples=n_samples)
# posterior_samples = posterior(*data["test"])
#
# k_hats = [psis_diagnostic(model, guide, *data["train"]) for _ in range(500)]
# k_hat = np.mean(k_hats)
#
# if k_hat < 0:
#     print(f"k hat = {k_hat:.3f} < 0: Wtf?")
# elif k_hat < 0.5:
#     print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
# elif k_hat < 0.7:
#     print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
# else:
#     print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")

df = pd.read_csv(
    "results/eightschools/number_of_flows_new.csv",
    # dtype={"type": "category", "n_flows": "Int64", "ELBO": float, "k_hat": float},
)

df_summary = df.groupby(["type", "n_flows"]).agg({"ELBO": [np.mean, np.std], "k_hat": [np.mean, np.std]}).reset_index()
with open("figures/eightschools/eightschools_error.tex", "w") as f:
    df_summary.to_latex(f, float_format="{:0.2f}".format)

fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
plt.subplots_adjust(wspace=0.3)
ax = axs[0]
k_hat_mf = df.query("type == 'Mean-field'")["k_hat"].values.item()
ax.axhline(k_hat_mf, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Mean-field",
    xy=(20, k_hat_mf),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
k_hat_fr = df.query("type == 'Full-rank'")["k_hat"].values.item()
ax.axhline(k_hat_fr, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Full-rank",
    xy=(22, k_hat_fr),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
ax.axhline(0.7, ls=":", linewidth=LINE_WIDTH, color="red", alpha=0.5)
ax.axhline(0.5, ls=":", linewidth=LINE_WIDTH, color="red", alpha=0.3)
ax.scatter(
    df.loc[df["type"] == "Planar", "n_flows"] + 0.1*np.random.randn(3*4),
    df.loc[df["type"] == "Planar", "k_hat"],
    color=PLANAR_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Planar", "n_flows"],
    df_summary.loc[df_summary["type"] == "Planar", ("k_hat", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Planar", ("k_hat", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=PLANAR_COLOR,
)
ax.annotate(
    text="Planar",
    xy=(7, 0.75),
    color=PLANAR_COLOR,
    size=LABEL_SIZE,
)
ax.scatter(
    df.loc[df["type"] == "Radial", "n_flows"] + 0.1*np.random.randn(3*4),
    df.loc[df["type"] == "Radial", "k_hat"],
    color=RADIAL_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Radial", "n_flows"],
    df_summary.loc[df_summary["type"] == "Radial", ("k_hat", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Radial", ("k_hat", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=RADIAL_COLOR,
)
ax.annotate(
    text="Radial",
    xy=(18, 0.73),
    color=RADIAL_COLOR,
    size=LABEL_SIZE,
)
ax.scatter(
    df.loc[df["type"] == "Inverse Autoregressive", "n_flows"] + 0.1*np.random.randn(3*4),
    df.loc[df["type"] == "Inverse Autoregressive", "k_hat"],
    color=IAF_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Inverse Autoregressive", "n_flows"],
    df_summary.loc[df_summary["type"] == "Inverse Autoregressive", ("k_hat", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Inverse Autoregressive", ("k_hat", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=IAF_COLOR,
)
ax.annotate(
    text="Inverse\nAutoregressive",
    xy=(10, 0.60),
    color=IAF_COLOR,
    size=LABEL_SIZE,
)
ax.set_xlabel("Number of flows")
ax.set_ylabel("$\\hat{k}$-statistic")
ax.set_xscale("log", base=2)
ax.set_xticks(df.loc[df["type"] == "Radial", "n_flows"])
finalize(ax)
adjust_spines(ax, ["left", "bottom"])

ax = axs[1]
ELBO_mf = df.query("type == 'Mean-field'")["ELBO"].values.item()
ax.axhline(ELBO_mf, ls=":", linewidth=LINE_WIDTH, color="grey")
# ax.annotate(
#     text="Mean-field",
#     xy=(50, ELBO_mf),
#     va="center",
#     backgroundcolor="white",
#     color="grey",
#     size=LABEL_SIZE,
# )
ELBO_fr = df.query("type == 'Full-rank'")["ELBO"].values.item()
ax.axhline(ELBO_fr, ls=":", linewidth=LINE_WIDTH, color="grey")
# ax.annotate(
#     text="Full-rank",
#     xy=(60, ELBO_fr),
#     va="center",
#     backgroundcolor="white",
#     color="grey",
#     size=LABEL_SIZE,
# )
ax.scatter(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "ELBO"],
    color=PLANAR_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Planar", "n_flows"],
    df_summary.loc[df_summary["type"] == "Planar", ("ELBO", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Planar", ("ELBO", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=PLANAR_COLOR,
)
ax.annotate(
    text="Planar",
    xy=(8, -33.5),
    color=PLANAR_COLOR,
    size=LABEL_SIZE,
)
ax.scatter(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "ELBO"],
    color=RADIAL_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Radial", "n_flows"],
    df_summary.loc[df_summary["type"] == "Radial", ("ELBO", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Radial", ("ELBO", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=RADIAL_COLOR,
)
ax.annotate(
    text="Radial",
    xy=(20, -34.5),
    color=RADIAL_COLOR,
    size=LABEL_SIZE,
)
ax.scatter(
    df.loc[df["type"] == "Inverse Autoregressive", "n_flows"],
    df.loc[df["type"] == "Inverse Autoregressive", "ELBO"],
    color=IAF_COLOR, alpha=0.3, edgecolors=ALMOST_BLACK,
)
ax.errorbar(
    df_summary.loc[df_summary["type"] == "Inverse Autoregressive", "n_flows"],
    df_summary.loc[df_summary["type"] == "Inverse Autoregressive", ("ELBO", "mean")],
    yerr=df_summary.loc[df_summary["type"] == "Inverse Autoregressive", ("ELBO", "std")], capsize=5,
    fmt=":o", markersize=5,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=IAF_COLOR,
)
ax.annotate(
    text="Inverse Autoregressive",
    xy=(4, -32.2),
    color=IAF_COLOR,
    size=LABEL_SIZE,
)
ax.set_xlabel("Number of flows")
ax.set_ylabel("ELBO")
ax.set_xscale("log", base=2)
ax.set_xticks(df.loc[df["type"] == "Radial", "n_flows"])
finalize(ax)
adjust_spines(ax, ["left", "bottom"])
fig.tight_layout()
save_plot(figure_path, "eightschools_replicates")







df = pd.read_csv(
    "results/eightschools/number_of_flows.csv",
    # dtype={"type": "category", "n_flows": "Int64", "ELBO": float, "k_hat": float},
)

fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
plt.subplots_adjust(wspace=0.3)
ax = axs[0]
k_hat_mf = df.query("type == 'Mean-field'")["k_hat"].values.item()
ax.axhline(k_hat_mf, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Mean-field",
    xy=(40, k_hat_mf),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
k_hat_fr = df.query("type == 'Full-rank'")["k_hat"].values.item()
ax.axhline(k_hat_fr, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Full-rank",
    xy=(50, k_hat_fr),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
ax.axhline(0.7, ls=":", linewidth=LINE_WIDTH, color='red', alpha=0.5)
ax.plot(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "k_hat"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    linewidth=LINE_WIDTH, color=PLANAR_COLOR,

)
ax.annotate(
    text="Planar",
    xy=(7, 0.73),
    color=PLANAR_COLOR,
    size=LABEL_SIZE,
)
ax.plot(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "k_hat"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=RADIAL_COLOR,
)
ax.annotate(
    text="Radial",
    xy=(45, 0.75),
    color=RADIAL_COLOR,
    size=LABEL_SIZE,
)
ax.plot(
    df.loc[df["type"] == "Inverse Autoregressive", "n_flows"],
    df.loc[df["type"] == "Inverse Autoregressive", "k_hat"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    # linestyle=":",
    linewidth=LINE_WIDTH,
    color=IAF_COLOR,
)
ax.annotate(
    text="Inverse\nAutoregressive",
    xy=(4, 0.47),
    color=IAF_COLOR,
    size=LABEL_SIZE,
)
ax.set_xlabel("Number of flows")
ax.set_ylabel("$\\hat{k}$-statistic")
ax.set_xscale("log", base=2)
ax.set_xticks(df.loc[df["type"] == "Radial", "n_flows"])
finalize(ax)
adjust_spines(ax, ["left", "bottom"])

ax = axs[1]
ELBO_mf = df.query("type == 'Mean-field'")["ELBO"].values.item()
ax.axhline(ELBO_mf, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Mean-field",
    xy=(50, ELBO_mf),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
ELBO_fr = df.query("type == 'Full-rank'")["ELBO"].values.item()
ax.axhline(ELBO_fr, ls=":", linewidth=LINE_WIDTH, color="grey")
ax.annotate(
    text="Full-rank",
    xy=(60, ELBO_fr),
    va="center",
    backgroundcolor="white",
    color="grey",
    size=LABEL_SIZE,
)
ax.plot(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "ELBO"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    # linestyle=":",
    linewidth=LINE_WIDTH, color=PLANAR_COLOR,
)
ax.annotate(
    text="Planar",
    xy=(8, -32),
    color=PLANAR_COLOR,
    size=LABEL_SIZE,
)
ax.plot(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "ELBO"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    # linestyle=":",
    linewidth=LINE_WIDTH, color=RADIAL_COLOR,
)
ax.annotate(
    text="Radial",
    xy=(32, -34.5),
    color=RADIAL_COLOR,
    size=LABEL_SIZE,
)
ax.plot(
    df.loc[df["type"] == "Inverse Autoregressive", "n_flows"],
    df.loc[df["type"] == "Inverse Autoregressive", "ELBO"],
    ":o", markersize=7, markeredgecolor=ALMOST_BLACK, markeredgewidth=.9,
    # linestyle=":",
    linewidth=LINE_WIDTH, color=IAF_COLOR,
)
ax.set_xlabel("Number of flows")
ax.set_ylabel("ELBO")
ax.set_xscale("log", base=2)
ax.set_xticks(df.loc[df["type"] == "Radial", "n_flows"])
finalize(ax)
adjust_spines(ax, ["left", "bottom"])
fig.tight_layout()
save_plot(figure_path, "eightschools")



with open("results/eightschools/eightschools_iaf_4_1.pkl", "rb") as f:
    results = pickle.load(f)
with open("results/eightschools/eightschools_hmc.pkl", "rb") as f:
    results_hmc = pickle.load(f)

# print(results["losses"][-500:].mean())
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(-results["losses"])
# ax.set(
#     xlabel="Iteration",
#     ylabel="ELBO",
#     xlim=[950, 1000],
#     ylim=[-30, -34],
# )

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(
    results_hmc["samples"]["tau"].log().cpu().detach().numpy(),
    results_hmc["samples"]["theta"][:, 0].cpu().detach().numpy(),
    label="NUTS",
    color="k",
    alpha=0.25,
)
ax.scatter(
    np.log(results["samples"]["tau"].squeeze()),
    results["samples"]["theta"][:, 0],
    label="VI with 4 Inverse Autoregressive flows",
    alpha=0.50,
    color=IAF_COLOR,
)
ax.set(
    xlabel="log $\\tau$",
    ylabel="$\\theta_1$",
    xlim=[-2, 3],
    ylim=[-20, 45],
    # title="Joints of $\\tau$ and $\\theta_1$",
)
ax.legend()
adjust_spines(ax, ["left", "bottom"])
finalize(ax)
# fig.subplots_adjust(left=0.25, right=0.75, bottom=0.2)
fig.tight_layout()
save_plot(figure_path, "log_tau_vs_theta")

plt.show()
