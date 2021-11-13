from distutils.spawn import find_executable
use_latex = False
if find_executable("pdflatex"):
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


def save_plot(location, name, use=None, axis_height="\\figureheight", axis_width="\\figurewidth"):
    full_path_without_ext = os.path.join(location, name)

    if use is None:
        use = ["pdf", "tikz", "pgf"]

    backend = None
    if use_latex:
        backend = "pgf"

    if "pgf" in use and use_latex:
        print(f"Saving {full_path_without_ext}.pgf.")
        plt.savefig(full_path_without_ext + ".pgf", backend=backend)
    if "pdf" in use:
        print(f"Saving {full_path_without_ext}.pdf.")
        plt.savefig(full_path_without_ext + ".pdf", backend=backend)
    if "tikz" in use:
        print(f"Saving {full_path_without_ext}.tex.")
        # tikzplotlib.clean_figure()
        tikzplotlib.save(full_path_without_ext + ".tex", axis_height=axis_height, axis_width=axis_width)
    if "png" in use:
        plt.savefig(full_path_without_ext + ".png", backend=backend)


import numpy as np
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio


def latexplots(height = 0, width = 16., fontsize = 12, pad= 2, heightmult=1.):
    fig_width_pt = width  # in cm, use with width=xcm in latex figure
    fig_width = fig_width_pt/2.54  # width in inches
    if height == 0:
        fig_height = fig_width*golden_mean    # height in inches
    else:
        fig_height = height/2.54
    fig_size =  [fig_width, fig_height*heightmult]

    print('fig size:', fig_size)
    params = {'backend': 'pdf',
              'pgf.texsystem'   :'pdflatex',
              'axes.labelsize'  :fontsize,
              'font.size'       :fontsize,
              'axes.titlesize'  :fontsize,
              'legend.fontsize' :fontsize-2,
              'xtick.labelsize' :fontsize-2,
              'ytick.labelsize' :fontsize-2,
              'xtick.major.pad' :pad,
              'xtick.minor.pad' :pad,
              'ytick.major.pad' :pad,
              'ytick.minor.pad' :pad,
              'figure.figsize'  :fig_size,
              'font.family'     :'serif',
              'text.usetex'     :True,
              'axes.grid'       :True,
              'axes.linewidth'  :0.5,
              'legend.numpoints':1,
              'legend.borderaxespad':0.3, # distance to axes edge
              'legend.borderpad':0.3,
              'legend.labelspacing':0.4,
              'legend.framealpha':1.0,
              'legend.frameon'   :True,
              'legend.handlelength':1.0,
              'legend.handletextpad':0.5,
              'legend.columnspacing':1.0,
              # 'axes.color_cycle': colors(8),
              'axes.formatter.useoffset': False,
              }
    plt.rcParams.update(params)
    plt.rcParams['text.latex.preamble'] = [
             r'\usepackage{siunitx}',
             r'\usepackage{helvet}',
             r'\usepackage[usenames]{color}',
             r'\usepackage{bm}',
    ]
    plt.rcParams["pgf.preamble"] = [
        r"\usepackage[utf8x]{inputenc}", # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
        ]
