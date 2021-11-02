import seaborn as sns
from src.problems.multivariate_normal import MultivariateNormal
from src.guides import mean_field
from src.experiments import train
from src.experiments.setup import setup
setup()

problem = MultivariateNormal()
data = problem.get_data()
sns.scatterplot(x=data["data"][:, 0], y=data["data"][:, 1]) \
    .set(
        xlabel="Age",
        ylabel="Number of deaths",
    )

model = problem.model
guide = mean_field(model)
train_result = train(model, guide, 5000, adam_params={"lr": 5e-3})
