# https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html
class PoissonRegression:
    def __init__(self):
        pass

    def poisson_model(self, X, y):
        w0 = pyro.sample("w0", dist.Normal(0,1))
        with pyro.plate("wk", K):
            wk = pyro.sample("wk", dist.Normal(0,1))
        with pyro.plate("yn", N):
            yn = torch.sample("yn", dist.Poisson(torch.exp(w0+X@w.T)), obs=y)

    def get_data(self):
        pass