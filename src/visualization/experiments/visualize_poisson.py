from src.visualization.setup import setup
setup()
from src.problems.poisson_regression import PoissonRegression
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib
import pickle
import numpy as np


def plot_summary(ax, x, s, interval=95, num_samples=100, sample_color='k', sample_alpha=0.4, interval_alpha=0.25,
                 color='r', legend=True, title="", plot_mean=True, plot_median=False, label="", seed=0):
    b = 0.5 * (100 - interval)

    lower = np.percentile(s, b, axis=0).T
    upper = np.percentile(s, 100 - b, axis=0).T

    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        lab = 'Median'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), median, label=lab, color=color, linewidth=4)

    if plot_mean:
        mean = np.mean(s, axis=0).T
        lab = 'Mean'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), mean, '--', label=lab, color=color, linewidth=4)
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha,
                    label='%d%% Interval' % interval)

    if num_samples > 0:
        np.random.seed(seed)
        idx_samples = np.random.choice(range(len(s)), size=num_samples, replace=False)
        ax.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);

    if legend:
        ax.legend(loc='best')

    if len(title) > 0:
        ax.title(title, fontweight='bold')


def plot_predictions(ax, x, s, num_samples=100, sample_color='k', sample_alpha=0.4, color='r', legend=False,
                     plot_median=False, plot_mean=True, seed=123, title=''):
    plot_summary(ax, x, s, color=color, interval=99, num_samples=0, interval_alpha=0.25, plot_mean=False,
                 plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=95, num_samples=0, sample_alpha=0.1, interval_alpha=0.35,
                 plot_mean=False, plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=75, interval_alpha=0.6, num_samples=num_samples,
                 sample_alpha=sample_alpha, plot_mean=False, plot_median=False, legend=legend, seed=seed,
                 sample_color=sample_color)

    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        ax.plot(x.ravel(), median, label='Median', color='k', linewidth=4, alpha=0.7)

    if plot_mean:
        mean = np.mean(s, axis=0).T
        ax.plot(x.ravel(), mean, '-', label='Mean', color='k', linewidth=4, alpha=0.7)

    if title:
        ax.set_title(title, fontweight='bold')


problem = PoissonRegression()
data = problem.get_data()

sns.scatterplot(
    x=data["age"],
    y=data["deaths"],
    color="black",
).set(
    xlabel="Age",
    ylabel="Number of deaths",
)
tikzplotlib.save("figures/poisson/data.tex")
# plt.show()



with open("results/poisson/poisson.pkl", "rb") as f:
    results = pickle.load(f)
prediction = results["predictive_samples"]


fig, ax = plt.subplots(1,1)
plot_predictions(ax, data["age"], prediction.detach().numpy(), num_samples=0, legend=True)
ax.scatter(data["age"], data["deaths"], c="k")
plt.savefig("figures/poisson/posterior_predictive.pgf")

plt.show()
