import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.setup import setup, save_plot
setup()

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
    "results/eightschools/number_of_flows.csv",
    # dtype={"type": "category", "n_flows": "Int64", "ELBO": float, "k_hat": float},
)

fig, axs = plt.subplots(ncols=2)
ax = axs[0]
ax.scatter(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "k_hat"],
)
ax.plot(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "k_hat"],
    linestyle="--", linewidth=2,
)
ax.annotate(
    text="Planar",
    xy=(12, 0.66),
)
ax.scatter(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "k_hat"],
)
ax.plot(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "k_hat"],
    linestyle="--", linewidth=2,
)
ax.annotate(
    text="Radial",
    xy=(37, 0.80),
)
k_hat_mf = df.query("type == 'Mean-field'")["k_hat"].values.item()
ax.axhline(k_hat_mf, ls='--', linewidth=2, color='grey')
ax.annotate(
    text="Mean-field",
    xy=(60, k_hat_mf),
    va="center",
    backgroundcolor='white',
    color="grey",
)
k_hat_fr = df.query("type == 'Full-rank'")["k_hat"].values.item()
ax.axhline(k_hat_fr, ls='--', linewidth=2, color='grey')
ax.annotate(
    text="Full-rank",
    xy=(70, k_hat_fr),
    va="center",
    backgroundcolor='white',
    color="grey",
)
ax.axhline(0.7, ls='--', linewidth=2, color='red')

ax = axs[1]
ax.scatter(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "ELBO"],
)
ax.plot(
    df.loc[df["type"] == "Planar", "n_flows"],
    df.loc[df["type"] == "Planar", "ELBO"],
    linestyle="--", linewidth=2,
)
ax.annotate(
    text="Planar",
    xy=(12, -35),
)
ax.scatter(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "ELBO"],
)
ax.plot(
    df.loc[df["type"] == "Radial", "n_flows"],
    df.loc[df["type"] == "Radial", "ELBO"],
    linestyle="--", linewidth=2,
)
ax.annotate(
    text="Radial",
    xy=(37, -35),
)
ELBO_mf = df.query("type == 'Mean-field'")["ELBO"].values.item()
ax.axhline(ELBO_mf, ls='--', linewidth=2, color='grey')
ax.annotate(
    text="Mean-field",
    xy=(60, ELBO_mf),
    va="center",
    backgroundcolor='white',
    color="grey",
)
ELBO_fr = df.query("type == 'Full-rank'")["ELBO"].values.item()
ax.axhline(ELBO_fr, ls='--', linewidth=2, color='grey')
ax.annotate(
    text="Full-rank",
    xy=(70, ELBO_fr),
    va="center",
    backgroundcolor='white',
    color="grey",
)
save_plot(figure_path, "eightschools")


with open("results/eightschools/eightschools.pkl", "rb") as f:
    results = pickle.load(f)
with open("results/eightschools/eightschools_hmc.pkl", "rb") as f:
    results_hmc = pickle.load(f)

# print(results["losses"][-500:].mean())

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
    label="VI with 16 Planar flows",
    alpha=0.25,
    color="blue",
)
ax.set(
    xlabel="log $\\tau$",
    ylabel="$\\theta_1$",
    xlim=[-2, 3],
    ylim=[-20, 45],
    # title="Joints of $\\tau$ and $\\theta_1$",
)
ax.legend()
save_plot(figure_path, "log_tau_vs_theta")

plt.show()
