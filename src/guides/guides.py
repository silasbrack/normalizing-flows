from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoNormalizingFlow
from functools import partial
from pyro.distributions.transforms import iterated

def mean_field(model):
    return AutoDiagonalNormal(model)

def full_rank(model):
    return AutoMultivariateNormal(model)

def normalizing_flow(model, num_flows, flow_type):
    transform_init = partial(iterated, num_flows, flow_type)
    return AutoNormalizingFlow(model, transform_init)
