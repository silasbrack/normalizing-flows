import torch
import numpy as np
import matplotlib.pyplot as plt

def _evaluate_pdf(exact_log_density, lims, point_density):
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], point_density),
                        np.linspace(lims[1][0], lims[1][1], point_density))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1))
    density = torch.exp(exact_log_density(z)).reshape(point_density, point_density)
    return density

def plot_pdf_image(exact_log_density, ax, lims=np.array([[-4, 4], [-4, 4]]), point_density=100,
                       name=None):
    density = _evaluate_pdf(exact_log_density, lims, point_density)
    ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap="coolwarm")
    if name is not None:
        plt.title(name, fontsize=22)

def plot_pdf_contours(dist, xmin=-4,xmax=4,ymin=-4,ymax=4,point_density=100):
    x = np.linspace(xmin, xmax, point_density)
    y = np.linspace(ymin, ymax, point_density)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    Z = torch.exp(dist.log_prob(torch.tensor(pos)))
    CS = plt.contour(X, Y, Z, levels=10, cmap="coolwarm")
