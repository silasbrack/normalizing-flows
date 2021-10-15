import seaborn as sns
from pyro.distributions.transforms import planar
from src.experiments.train import train
from src.guides import normalizing_flow
from src.problems.poisson_regression import PoissonRegression
from src.experiments.setup import set_seeds
set_seeds()

problem = PoissonRegression()
data = problem.get_data()
sns.scatterplot(x=data["age"], y=data["deaths"]) \
    .set(
        xlabel="Age",
        ylabel="Number of deaths",
    )

model = problem.poisson_model
guide = normalizing_flow(model, flow_type=planar, num_flows=4)
train_result = train(model, guide, 5000, adam_params={"lr": 5e-3})
