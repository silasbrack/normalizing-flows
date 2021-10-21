# Variational inference using normalizing flows

## Structure

├── LICENSE
├── README.md           <- The top-level README for developers using this project.
├── figures             <- Figures generated from experiment results in tikz, pgf, eps or pdf.
│   ├── energy
│   ├── experiment1
│   ├── experiment2
│   └── ...             <- Name taken from src/problems/
├── results             <- Results from experiments in numpy array / dict format.
│   ├── energy
│   ├── experiment1
│   ├── experiment2
│   └── ...             <- Name taken from src/problems/
├── src                 <- Source code.
│   ├── experiments     <- Running experiments to generate data to results/.
│   ├── guides          <- Guides in Pyro (where we define variational approximations).
│   ├── problems        <- Problems, including data and models.
│   └── visualization   <- Visualization, e.g., helper functions and experiments.
│       └── experiments <- Visualizing experiments to generate figures to figures/.
├── data
│   ├── external        <- Data from third party sources.
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump.
│
└── tox.ini             <- tox file with settings for running tox; see tox.readthedocs.io
