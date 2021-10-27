import torch
import pyro
from pyro import distributions as dist
from sklearn.preprocessing import scale
from torch import nn
from pyro.nn import PyroSample, PyroModule


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        rate = self.linear(x).squeeze(-1).exp()
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Poisson(rate), obs=y)
            return obs

# Let's set-up the design matrix for the training data and the inputs for predictions
def design_matrix(x): return torch.column_stack((torch.ones(len(x)), x))


# https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html
class PoissonRegression:
    def __init__(self):
        self.dim = 2

    poisson_model = BayesianRegression(1, 1)

    # def poisson_model(self, x, y=None):
    #     # Priors
    #     kappa2 = pyro.sample("kappa2", dist.LogNormal(0, 1))
    #     with pyro.plate("wk", self.dim):
    #         w = pyro.sample("w", dist.Normal(0, kappa2))
    #
    #     # Likelihood
    #     with pyro.plate("yn", x.shape[0]):
    #         obs = pyro.sample("obs", dist.Poisson(torch.exp(x @ w.T)), obs=y)
    #         return obs

    @staticmethod
    def get_data():
        # The input covariate is age
        age = torch.arange(35, 65).float()

        # The output is the number of deaths in the study popular for each age
        deaths = torch.tensor(
            [3, 1, 3, 2, 2, 4, 4, 7, 5, 2, 8, 13, 8, 2, 7, 4, 7, 4, 4, 11, 11, 13, 12, 12, 19, 12, 16, 12, 6,
             10]).float()

        # prep data
        x = torch.tensor(scale(age))
        x = x[:, None]
        # x = design_matrix(x)
        x = x.float()

        return {"design_matrix": x, "age": age, "deaths": deaths}
