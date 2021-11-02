import pandas as pd
import pyro
from pyro.distributions.transforms import planar
from src.experiments.train import train
from src.guides import normalizing_flow
from src.problems.poisson_regression import PoissonRegression
from src.experiments.setup import setup
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
import pickle
device = setup()


results_path = "results/poisson/"
model_path = "models/poisson/"

problem = PoissonRegression(device=device)
data = problem.get_data()

model = problem.model
guide = normalizing_flow(model, flow_type=planar, num_flows=4)
train_result = train(
    data["train"],
    model,
    guide,
    epochs=1000,
    gradient_mc_samples=512,
    adam_params={"lr": 1e-2},
)

# Sample from posterior
posterior_predictive = Predictive(model=model, guide=guide, num_samples=1024)
predictive_samples = posterior_predictive.get_samples(*data["test"])
predictive_samples = {k: v.cpu().detach().numpy() for k, v in predictive_samples.items()}
train_result["samples"] = predictive_samples

# Calculate k hat statistic
# k_hat = psis_diagnostic(model, guide, *data["train"])
k_hats = [psis_diagnostic(model, guide, *data["train"]) for _ in range(500)]

df_k = pd.DataFrame(data=k_hats, columns=["k_hat"])
df_k["<0.5"] = df_k["k_hat"] <= 0.5
df_k["<0.7"] = (df_k["k_hat"] > 0.5) & (df_k["k_hat"] <= 0.7)
df_k[">0.7"] = df_k["k_hat"] > 0.7
df_k = df_k.agg(["mean"])
print(df_k)

k_hat = df_k.loc["mean", "k_hat"]
if k_hat < 0.5:
    print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
elif k_hat < 0.7:
    print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
else:
    print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")

train_result["k_hat"] = k_hat

# ax = sns.scatterplot(x=dist.Normal(0., .1).sample(torch.Size([len(k_hats)])), y=k_hats)
# ax.axhline(0.7, ls='--', linewidth=3, color='r', label="0.7 threshold: poor fit")
# ax.axhline(0.5, ls='-.', linewidth=3, color='r', alpha=0.5, label="0.5 threshold: possibly poor fit")
# ax.set(
#     title="Jittered scatterplot of $\\hat{k}$",
#     ylabel="$\\hat{k}$"
# )
# plt.show()

file_name = "poisson"
with open(results_path + file_name + ".pkl", "wb") as f:
    pickle.dump(train_result, f)
pyro.get_param_store().save(model_path + file_name + ".save")
