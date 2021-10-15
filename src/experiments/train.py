from pyro.optim import Adam
import pyro
from pyro.infer import Trace_ELBO, SVI
from collections import defaultdict
import numpy as np
from pyro import poutine
import tqdm
import seaborn as sns


def train(data, model, guide, epochs, gradient_mc_samples=1, adam_params={"lr": 3e-4}, plot=True):
    optim = Adam(adam_params)
    loss = Trace_ELBO(num_particles=gradient_mc_samples) # Have to fix stuff to use vectorize_particles=True
    svi = SVI(model, guide, optim, loss=loss)
    pyro.clear_param_store()

    gradient_norms = defaultdict(list)
    register_gradient_hooks(svi, gradient_norms, [*data])

    losses = np.zeros(epochs)
    for i in tqdm.tqdm(range(epochs)):
        losses[i] = svi.step(*data)

    if plot:
        sns.lineplot(data=-losses) \
            .set(
                xlabel="Iteration",
                ylabel="ELBO",
            )

    return {"svi": svi, "losses": losses, "gradient_norms": gradient_norms}


def register_gradient_hooks(svi, gradient_norms, svi_args=[]):
    # Register hooks to monitor gradient norms.
    svi.step(*svi_args)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))


# KL annealing, credit to https://pyro.ai/examples/custom_objectives.html#Example:-KL-Annealing
def simple_elbo_kl_annealing(model, guide, *args, **kwargs):
    """
    :param model:
    :param guide:
    :param args:
    :param kwargs:
    :return:

    svi = SVI(model, guide, optim, loss=simple_elbo_kl_annealing)
    svi.step(other_args, annealing_factor=0.2, latents_to_anneal=["my_latent"])
    """

    # get the annealing factor and latents to anneal from the keyword
    # arguments passed to the model and guide
    annealing_factor = kwargs.pop('annealing_factor', 1.0)
    latents_to_anneal = kwargs.pop('latents_to_anneal', [])
    # run the guide and replay the model against the guide
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    elbo = 0.0
    # loop through all the sample sites in the model and guide trace and
    # construct the loss; note that we scale all the log probabilities of
    # samples sites in `latents_to_anneal` by the factor `annealing_factor`
    for site in model_trace.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo + factor * site["fn"].log_prob(site["value"]).sum()
    for site in guide_trace.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo - factor * site["fn"].log_prob(site["value"]).sum()
    return -elbo
