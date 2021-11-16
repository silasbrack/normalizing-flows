import random
import numpy as np
import torch
import pyro


def setup(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.enable_validation(True)

    torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device
