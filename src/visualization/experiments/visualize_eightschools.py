import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
import pandas as pd
import pyro
from pyro.distributions.transforms import planar
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
from src.guides import normalizing_flow
from src.problems import EightSchools
import arviz as az
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
print(df)

df_flows = df.query("type in ['Planar', 'Radial']")
df_other = df.query("type in ['Mean-field', 'Full-rank']")

df_other.loc[df_other["type"] == "Mean-field", "n_flows"] = 6
df_other.loc[df_other["type"] == "Full-rank", "n_flows"] = 15
p = (
    ggplot(df_flows, aes(x="n_flows", y="k_hat", color="factor(type)")) +
    geom_point() + geom_line(linetype="dashed") +
    geom_text(aes(label="type"), y=[0.79, 0.91], data=df_flows.query("n_flows == 8")) +
    geom_hline(aes(yintercept="k_hat"), data=df_other, linetype="dashed", color="grey") +
    geom_label(aes(label="type"), data=df_other, label_size=0, color="grey") +
    geom_hline(yintercept=0.7, linetype="dashed", color="red") +
    scale_x_continuous(trans="log2") +
    labs(x="Number of flows", y="k-hat statistic") +
    theme_seaborn(style="ticks") +
    theme(legend_position="none")
)
fig = p.draw()
fig.show()
p.save(figure_path + "k_hat.pdf", width=4, height=4)

df_other.loc[df_other["type"] == "Mean-field", "n_flows"] = 12
df_other.loc[df_other["type"] == "Full-rank", "n_flows"] = 25
p = (
    ggplot(df_flows, aes(x="n_flows", y="ELBO", color="factor(type)")) +
    geom_point() + geom_line(linetype="dashed") +
    geom_text(aes(label="type"), y=[-32.1, -36.35], data=df_flows.query("n_flows == 16")) +
    geom_hline(aes(yintercept="ELBO"), data=df_other, linetype="dashed", color="grey") +
    geom_label(aes(label="type"), data=df_other, label_size=0, color="grey") +
    scale_x_continuous(trans="log2") +
    labs(x="Number of flows", y="ELBO") +
    theme_seaborn(style="ticks") +
    theme(legend_position="none")
)
fig = p.draw()
fig.show()
p.save(figure_path + "elbo.pdf", width=4, height=4)
# ax = sns.scatterplot(
#     data=df_flows,
#     x="n_flows",
#     y="k_hat",
#     hue="type",
#     s=75,
# )
# ax = sns.lineplot(
#     data=df_flows,
#     x="n_flows",
#     y="k_hat",
#     hue="type",
#     linestyle="--", linewidth=3,
#     ax=ax,
# )
# ax.axhline(df_other.query("type == 'Mean-field'")["k_hat"].values, ls='--', linewidth=3, color='grey')
# ax.axhline(df_other.query("type == 'Full-rank'")["k_hat"].values, ls='--', linewidth=3, color='grey')
# ax.get_legend().remove()
# save_plot(figure_path, "number_of_flows")


with open("results/eightschools/eightschools.pkl", "rb") as f:
    results = pickle.load(f)
with open("results/eightschools/eightschools_hmc.pkl", "rb") as f:
    results_hmc = pickle.load(f)

# print(results["losses"][-500:].mean())

plt.figure(figsize=(4, 4))
ax = sns.scatterplot(
    x=results_hmc["samples"]["tau"].log().cpu().detach().numpy(),
    y=results_hmc["samples"]["theta"][:, 0].cpu().detach().numpy(),
    label="NUTS",
    color="k",
    alpha=0.25,
)
ax = sns.scatterplot(
    x=np.log(results["samples"]["tau"].squeeze()),
    y=results["samples"]["theta"][:, 0],
    label="VI with 16 Planar flows",
    alpha=0.25,
    color="blue",
    ax=ax,
)
ax.set(
    xlabel="log $\\tau$",
    ylabel="$\\theta_1$",
    xlim=[-2, 3],
    ylim=[-20, 45],
    # title="Joints of $\\tau$ and $\\theta_1$",
)
save_plot(figure_path, "log_tau_vs_theta")

plt.show()
