from plotnine import *
from src.problems.energy import EnergyPosteriorProblem
from pyro.distributions.transforms import planar, radial
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
sns.set_theme(context="paper", style="ticks")
import matplotlib.pyplot as plt

final_elbos = {"Planar": [], "Radial": []}
idx = pd.MultiIndex.from_product([['Planar','Radial'], [2,8,32]])
df : pd.DataFrame = pd.DataFrame(np.random.randn(4,6), columns=idx, index=["U1", "U2", "U3", "U4"])

for dist_name in ["U1", "U2", "U3", "U4"]:
    for name in ["Planar", "Radial"]:
        for n_flows in [2, 8, 32]:
            with open("results/energy/" + f"{dist_name}_{name.lower()}_{n_flows}.pkl", "rb") as f:
                results = pickle.load(f)

            flow_samples = results["samples"]
            intermediate_samples = results["intermediate_samples"]
            losses = results["losses"]
            average_final_elbo = -np.mean(losses[-1000:])
            df.loc[dist_name][name][n_flows] = average_final_elbo

            if dist_name == "U1":
                final_elbos[name].append(average_final_elbo)

df.index = ["$U_1(z)$", "$U_2(z)$", "$U_3(z)$", "$U_4(z)$"]
df = df.reset_index().rename(columns={"index": "type"}) \
       .melt(id_vars="type", value_name="ELBO", var_name=["flow", "n_flows"]) \
       .assign(idx=lambda df: df["n_flows"].map({2: 0, 8: 1, 32: 2}))

g = (
    sns.FacetGrid(df, col="type",  col_wrap=2, sharey=False, height=2.5, aspect=3/2.5)
    .map(sns.scatterplot, "idx", "ELBO", "flow", palette=["r", "b"], s=30)
    .map(sns.lineplot, "idx", "ELBO", "flow", palette=["r", "b"], linewidth=2, linestyle="--")
    .set_titles(col_template="{col_name}", row_template="{row_name}")
    .add_legend()
    .set_axis_labels("Number of flows", "ELBO")
)
sns.despine()
plt.show()

p = (
    ggplot(df, aes(x="n_flows", y="ELBO", color="flow"))
    + geom_point(size=2)
    + geom_line(linetype='dashed', size=1.5)
    + facet_wrap("~type", scales="free")
    # + theme(subplots_adjust={'hspace': 0.25, "wspace": 0.25})
)
print(p)
