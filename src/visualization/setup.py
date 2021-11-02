import matplotlib as mpl
mpl.rcParams.update({"pgf.texsystem": "pdflatex"})
import seaborn as sns
def setup():
    sns.set_theme(
        context="paper",
        style="ticks",  # Using white breaks tikzplotlib ticks, use ticks or darkgrid
        font="serif",
        rc={
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
