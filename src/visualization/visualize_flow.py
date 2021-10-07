from src.problems.nf import NormalizingFlow


def visualize_flow_as_gif(model: NormalizingFlow, to_save: bool):
    path_to_visualizations = "figures/"
    _save_to_gif(None, path_to_visualizations, "model.gif")


def _save_to_gif(animation, path_to_visualizations, filename):
    return animation, path_to_visualizations, filename
