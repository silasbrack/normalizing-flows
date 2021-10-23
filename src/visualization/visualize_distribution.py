import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pdf(dist, ax=None, x_min=-4, x_max=4, y_min=-4, y_max=4, point_density=100, how="contourf"):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x = np.linspace(x_min, x_max, point_density)
    y = np.linspace(y_min, y_max, point_density)
    x_grid, y_grid = np.meshgrid(x, y)
    pos = np.empty(x_grid.shape + (2,))
    pos[:, :, 0] = x_grid
    pos[:, :, 1] = y_grid
    z = torch.exp(dist.log_prob(torch.tensor(pos)))
    if how == "contourf":
        ax.contourf(x_grid, y_grid, z, levels=10)
    elif how == "contour":
        ax.contour(x_grid, y_grid, z, levels=10)
    elif how == "heatmap":
        sns.heatmap(
            data=z,
            cbar=False,
            ax=ax
        )
        ax.set(
            xticks=[],
            yticks=[],
        )
    else:
        raise Exception("Parameter 'how' must be in values ['contourf', 'contour','heatmap'].")
    return ax


def plot_samples(dist=None, samples=None, n=500, title="", ax=None, shade=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if dist is None and samples is None:
        raise ValueError('Expected either dist or samples args')
    elif samples is None:
        samples = dist.sample(torch.Size([n])).numpy()
    ax = sns.kdeplot(
        x=samples[:, 0],
        y=samples[:, 1],
        n_levels=10,
        shade=shade,
        cmap=sns.color_palette("rocket", as_cmap=True),
        ax=ax
    )
    ax.set(
        xlim=(-4, 4),
        ylim=(-4, 4),
        title=title,
    )
    return ax, samples
