import argparse
import pandas as pd
import pyro
from pyro.infer import NUTS, MCMC
from src.problems.eightschools import EightSchools
from src.experiments.setup import setup
import pickle
device = setup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hamiltonian Monte Carlo on the eight schools problem"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        metavar="N",
        help="number of samples to sample from MCMC (default: 1024)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1024,
        metavar="N",
        help="number MCMC warmup samples (default: 300)",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default='results/eightschools/',
        help="Path to results (default: results/eightschools/)",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="eightschools_hmc",
        help="Name of output files (default: eightschools_hmc)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    problem = EightSchools(device=device)
    data = problem.get_data()
    model = problem.model

    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps)
    mcmc.run(*data["train"])
    nuts_samples = mcmc.get_samples()
    train_result = {"samples": nuts_samples}

    with open(args.results_path + args.file_name + ".pkl", "wb") as f:
        pickle.dump(train_result, f)


if __name__ == "__main__":
    main()
