import random
import numpy as np
import torch
import pyro


def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.enable_validation(True)
