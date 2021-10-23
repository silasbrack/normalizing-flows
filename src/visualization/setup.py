import matplotlib as mpl
mpl.rcParams.update({"pgf.texsystem": "pdflatex"})
import seaborn as sns
def setup():
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
