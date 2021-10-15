import torch
import pyro
from pyro import distributions as dist
from sklearn.preprocessing import scale


# Let's set-up the design matrix for the training data and the inputs for predictions
def design_matrix(x): return torch.column_stack((torch.ones(len(x)), x))


# https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html
class PoissonRegression:
    def __init__(self):
        self.dim = 2

    def poisson_model(self, x, obs=None):
        # Priors
        kappa2 = pyro.sample("kappa2", dist.LogNormal(0, 1))
        with pyro.plate("wk", self.dim):
            w = pyro.sample("w", dist.Normal(0, kappa2)).double()

        # Likelihood
        with pyro.plate("yn", x.shape[0]):
            y = pyro.sample("y", dist.Poisson(torch.exp(x @ w.T)), obs=obs)
            return y

    @staticmethod
    def get_data():
        # The input covariate is age
        age = torch.arange(35, 65).double()

        # The output is the number of deaths in the study popular for each age
        deaths = torch.tensor(
            [3, 1, 3, 2, 2, 4, 4, 7, 5, 2, 8, 13, 8, 2, 7, 4, 7, 4, 4, 11, 11, 13, 12, 12, 19, 12, 16, 12, 6,
             10]).double()

        # prep data
        x = scale(age)
        x = x[:, None]
        x = design_matrix(x).double()

        return {"design_matrix": x, "ages": age, "deaths": deaths}
