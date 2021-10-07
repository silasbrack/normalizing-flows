from src.problems import multivariate_normal
from src.guides import mean_field
from src.experiments import train

model = multivariate_normal.model
guide = mean_field(model)
train_result = train(model, guide, 5000, adam_params={"lr": 5e-3})


