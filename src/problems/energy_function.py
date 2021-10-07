import torch
from pyro import distributions as dist

class EnergyDistribution(dist.TorchDistribution):
    def __init__(self, type):
        super().__init__(validate_args=False)

        self.type = type

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

        if self.type == "U1":
            self._energy_fn = self.U1
        elif self.type == "U2":
            self._energy_fn = self.U2
        elif self.type == "U3":
            self._energy_fn = self.U3
        elif self.type == "U4":
            self._energy_fn = self.U4
        else:
            raise BaseException("Distribution not found.")

        self._event_shape = torch.Size((2,))

    def log_prob(self, z):
        if len(z.shape) == 1:
            x, y = z[0], z[1]
        elif len(z.shape) == 2:
            x, y = z[: ,0], z[: ,1]
        elif len(z.shape) == 3:
            x, y = z[: ,: ,0], z[: ,: ,1]
        else:
            raise(BaseException("Error in log_prob dim: len(z.shape) > 3!"))
        return -self._energy_fn(x, y)


def model(x, p_z):
    # z ~ p(x,z) = p(z)
    with pyro.plate("data", x.shape[0]):
        z = pyro.sample("z", p_z)
        pyro.sample("x", p_z, obs=x)

def get_distribution(type):
    return EnergyDistribution(type)