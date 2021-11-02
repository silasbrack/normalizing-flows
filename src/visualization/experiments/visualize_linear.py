



# Create summary for latent variables
data_dict = {
    k: np.expand_dims(v.squeeze().cpu().detach().numpy(), 0)
    for k, v in predictive_samples.items()
    if "obs" not in k and "y" not in k
}
az_data = az.dict_to_dataset(data_dict)
summary = az.summary(az_data, round_to=4, kind="stats")

real_values = torch.cat((problem.true_params["alpha"].reshape(-1),
                         problem.true_params["beta"],
                         problem.true_params["sigma"].reshape(-1)))
summary["real"] = real_values
summary["in_bounds"] = (summary["real"] <= summary["hdi_97%"]) & (summary["real"] >= summary["hdi_3%"])

print(summary)

# Plot latent variable histograms
ax = sns.histplot(predictive_samples["sigma"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["sigma"], 0, 1000, c="r")
ax.set(title="Posterior samples $\\sigma$")
plt.show()

ax = sns.histplot(predictive_samples["alpha"].squeeze().cpu().detach().numpy())
ax.axvline(problem.true_params["alpha"], 0, 1000, c="r")
ax.set(title="Posterior samples $\\alpha$")
plt.show()
