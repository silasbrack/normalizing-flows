import torch
import pyro
from pyro import distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization.visualize_distribution import plot_samples


class EnergyDistribution(dist.TorchDistribution):
    def __init__(self, dist_type):
        super().__init__(validate_args=False)

        self.dist_type = dist_type
        self.dim = 2

        w1 = lambda x: torch.sin((2 * np.pi * x) / 4)
        w2 = lambda x: 3 * torch.exp(-(((x - 1) / 0.6) ** 2) / 2)
        w3 = lambda x: 3 * 1 / (1 + torch.exp(- ((x - 1) / 0.3)))

        self.U1 = lambda x, y: (((torch.sqrt(x ** 2 + y ** 2) - 2) / 0.4) ** 2) / 2 - torch.log(
            1e-15 + (torch.exp(-(((x - 2) / 0.6) ** 2) / 2) + torch.exp(-(((x + 2) / 0.6) ** 2) / 2)))
        self.U2 = lambda x, y: (((y - w1(x)) / 0.4) ** 2) / 2
        self.U3 = lambda x, y: - torch.log(1e-15 + torch.exp(-(((y - w1(x)) / 0.35) ** 2) / 2) + torch.exp(
            -(((y - w1(x) + w2(x)) / 0.35) ** 2) / 2))
        self.U4 = lambda x, y: - torch.log(1e-15 + torch.exp(-(((y - w1(x)) / 0.4) ** 2) / 2) + torch.exp(
            -(((y - w1(x) + w3(x)) / 0.35) ** 2) / 2))

        if self.dist_type == "U1":
            self._energy_fn = self.U1
        elif self.dist_type == "U2":
            self._energy_fn = self.U2
        elif self.dist_type == "U3":
            self._energy_fn = self.U3
        elif self.dist_type == "U4":
            self._energy_fn = self.U4
        else:
            raise Exception("Distribution not found.")

        self._event_shape = torch.Size((2,))

    def log_prob(self, z):
        if len(z.shape) == 1:
            x, y = z[0], z[1]
        elif len(z.shape) == 2:
            x, y = z[:, 0], z[:, 1]
        elif len(z.shape) == 3:
            x, y = z[:, :, 0], z[:, :, 1]
        else:
            raise Exception("Error in log_prob dim: len(z.shape) > 3!")
        return -self._energy_fn(x, y)


class EnergyPosteriorProblem:  # nn.Module
    def __init__(self,
                 dist_name,
                 nfs,
                 dim=2,
                 base_dist=lambda dim: dist.Normal(torch.zeros(dim), torch.ones(dim)),
                 device=torch.device("cpu")
                 ):
        self.uuid = np.random.randint(low=0, high=10000, size=1)[0]
        self.dim = dim
        self.base_dist = base_dist(dim)
        self.n_flows = len(nfs)
        self.nfs = nfs
        self.nf_dist = dist.TransformedDistribution(self.base_dist, self.nfs)
        self._register()

        self.device = device

        self.dist_name = dist_name
        self.target_dist = EnergyPosteriorProblem.get_dist(dist_name)

    def _register(self):
        for f in range(self.n_flows):
            nf_module = pyro.module("%d_nf_%d" % (self.uuid, f), self.nfs[f])

    def model(self, x):
        # z ~ p(x,z) = p(z)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("z", self.target_dist)
            pyro.sample("x", self.target_dist, obs=x)

    def guide(self, x):
        # z ~ q(z|x)
        self._register()
        with pyro.plate("data", x.shape[0]):
            pyro.sample("z", self.nf_dist)

    def log_prob(self, z):
        # log q(z|x)
        return self.nf_dist.log_prob(z)

    def sample(self, n):
        return self.nf_dist.sample(torch.Size([n]))

    def train(self, epochs, gradient_mc_samples=1, n_samples=256, adam_params={"lr": 3e-4}, plot=True):
        optim = Adam(adam_params)
        loss = Trace_ELBO(num_particles=gradient_mc_samples, vectorize_particles=True)  # Have to fix stuff to use vectorize_particles=True
        svi = SVI(self.model, self.guide, optim, loss=loss)
        pyro.clear_param_store()

        intermediate_samples = {}

        losses = np.zeros(epochs)
        for i in tqdm.tqdm(range(epochs)):
            z0 = self.sample(n_samples).to(self.device)
            losses[i] = svi.step(z0)

            if plot and i % 1000 == 0:
                _, samples = self.plot_flow_samples(title=f"$z_{{{self.n_flows}}}$ for iteration {i}")
                intermediate_samples[i] = samples
                plt.show()

        if plot:
            sns.lineplot(data=-losses) \
                .set(
                xlabel="Iteration",
                ylabel="ELBO",
            )
            plt.show()

        return {"losses": losses, "intermediate_samples": intermediate_samples}

    def plot_flow_samples(self, f=None, title="", ax=None):
        if f is None:
            f = self.n_flows
        intermediate_nf = dist.TransformedDistribution(self.base_dist, self.nfs[:f])
        ax, samples = plot_samples(intermediate_nf, title=title, ax=ax)
        return ax, samples

    @staticmethod
    def get_dist(dist_name):
        return EnergyDistribution(dist_name)
