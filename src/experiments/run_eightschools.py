import argparse
import pandas as pd
import pyro
from pyro.distributions.transforms import planar, radial
from pyro.infer import Predictive
from pyro.infer.importance import psis_diagnostic
from src.experiments.train import train
from src.guides import normalizing_flow
from src.problems.eightschools import EightSchools
from src.experiments.setup import setup
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
device = setup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eight schools problem"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        metavar="N",
        help="number of epochs to train (default: 10000)",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default='results/eightschools/',
        help="Path to results (default: results/eightschools/)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/eightschools/",
        help="Path to model file (default: models/eightschools/)",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="eightschools",
        help="Name of output files (default: eightschools)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate of ADAM optimizer (default: 1e-2)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=256,
        help="Number of Monte Carlo ELBO gradient estimation samples (default: 256)",
    )
    parser.add_argument(
        "--flow-type",
        type=str,
        default="planar",
        help="Type of flow, planar or radial (default: planar)",
    )
    parser.add_argument(
        "--num-flows",
        type=int,
        default=16,
        help="Number of flows (default: 16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    problem = EightSchools(device=device)
    data = problem.get_data()

    flow_type = {"planar": planar, "radial": radial}[args.flow_type]

    model = problem.model
    guide = normalizing_flow(model, flow_type=flow_type, num_flows=args.num_flows)
    train_result = train(
        data["train"],
        model,
        guide,
        epochs=args.epochs,
        gradient_mc_samples=args.mc_samples,
        adam_params={"lr": args.lr},
    )

    # sns.lineplot(x=range(args.epochs), y=train_result["losses"])
    plt.show()

    # Sample from posterior
    posterior_predictive = Predictive(model=model, guide=guide, num_samples=1024)
    predictive_samples = posterior_predictive.get_samples(*data["test"])
    predictive_samples = {k: v.cpu().detach().numpy() for k, v in predictive_samples.items()}
    train_result["samples"] = predictive_samples

    # Calculate k hat statistic
    k_hats = [psis_diagnostic(model, guide, *data["train"]) for _ in range(500)]

    df_k = pd.DataFrame(data=k_hats, columns=["k_hat"])
    df_k["<0.5"] = df_k["k_hat"] <= 0.5
    df_k["<0.7"] = (df_k["k_hat"] > 0.5) & (df_k["k_hat"] <= 0.7)
    df_k[">0.7"] = df_k["k_hat"] > 0.7
    df_k = df_k.agg(["mean"])
    print(df_k)

    k_hat = df_k.loc["mean", "k_hat"]
    if k_hat < 0.5:
        print(f"k hat = {k_hat:.3f} < 0.5: VI approximation is reliable.")
    elif k_hat < 0.7:
        print(f"k hat = {k_hat:.3f} < 0.7: VI approximation is moderately reliable.")
    else:
        print(f"k hat = {k_hat:.3f} > 0.7: VI approximation is NOT reliable.")

    train_result["k_hat"] = k_hat

    with open(args.results_path + args.file_name + ".pkl", "wb") as f:
        pickle.dump(train_result, f)
    pyro.get_param_store().save(args.model_path + args.file_name + ".save")

if __name__ == "__main__":
    main()
