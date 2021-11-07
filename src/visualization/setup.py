from distutils.spawn import find_executable
use_latex = False
if find_executable("latex"):
    use_latex = True

rc = {}
if use_latex:
    import matplotlib as mpl
    mpl.rcParams.update({"pgf.texsystem": "pdflatex"})
    rc = {"text.usetex": True, "pgf.rcfonts": False}

import matplotlib.pyplot as plt
import tikzplotlib
import os
import seaborn as sns  # Need to import after setting mpl.rcParams
def setup():
    sns.set_theme(
        context="paper",
        style="ticks",  # Using white breaks tikzplotlib ticks
        font="serif",
        rc=rc,
    )
    return use_latex


def save_plot(location, name, use=["pdf", "tikz", "pgf"], axis_height = '\\figureheight', axis_width = '\\figurewidth'):
    full_path_without_ext = os.path.join(location, name)

    backend = None
    if use_latex:
        backend = "pgf"

    if "pgf" in use and use_latex:
        # print(f"Saving {full_path_without_ext}.pgf.")
        plt.savefig(full_path_without_ext + ".pgf", backend=backend)
    if "pdf" in use:
        # print(f"Saving {full_path_without_ext}.pdf.")
        plt.savefig(full_path_without_ext + ".pdf", backend=backend)
    if "tikz" in use and use_latex:
        # print(f"Saving {full_path_without_ext}.tex.")
        tikzplotlib.clean_figure()
        tikzplotlib.save(full_path_without_ext + ".tex", axis_height=axis_height, axis_width=axis_width)
