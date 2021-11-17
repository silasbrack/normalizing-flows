#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2017, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

import numpy as np
from matplotlib import rcParams

# Project theme
PLANAR_COLOR = np.array([0.298039215686275, 0.447058823529412, 0.690196078431373])
RADIAL_COLOR = np.array([0.866666666666667, 0.517647058823529, 0.32156862745098])
IAF_COLOR = np.array([0.333333333333333, 0.658823529411, 0.407843137254902])

# Constants
SINGLE_FIG_SIZE = (6, 4)
BAR_WIDTH = 0.6
TICK_SIZE = 15
XLABEL_PAD = 10
LABEL_SIZE = 14
TITLE_SIZE = 16
LEGEND_SIZE = 12
LINE_WIDTH = 2
LIGHT_COLOR = '0.8'
LIGHT_COLOR_V = np.array([float(LIGHT_COLOR) for i in range(3)])
DARK_COLOR = '0.4'
DARK_COLOR_V = np.array([float(DARK_COLOR) for i in range(3)])
ALMOST_BLACK = '0.125'
ALMOST_BLACK_V = np.array([float(ALMOST_BLACK) for i in range(3)])
ACCENT_COLOR_1 = np.array([255., 145., 48.]) / 255.

# Configuration
# rcParams['text.usetex'] = True #Let TeX do the typsetting
# rcParams['pdf.use14corefonts'] = True
# rcParams['ps.useafm'] = True
# rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
rcParams['font.family'] = 'serif'  # ... for regular text
# rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica Neue',
#                                'HelveticaNeue']  # , Avant Garde, Computer Modern Sans serif' # Choose a nice font here
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = ALMOST_BLACK
rcParams['axes.unicode_minus'] = False

rcParams['xtick.major.pad'] = '8'
rcParams['axes.edgecolor'] = ALMOST_BLACK
rcParams['axes.labelcolor'] = ALMOST_BLACK
rcParams['lines.color'] = ALMOST_BLACK
rcParams['xtick.color'] = ALMOST_BLACK
rcParams['ytick.color'] = ALMOST_BLACK
rcParams['text.color'] = ALMOST_BLACK
rcParams['lines.solid_capstyle'] = 'butt'

# rcParams['axes.labelweight'] = 'bold'

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# Make a themed colormap
r, g, b = ACCENT_COLOR_1
cdict = {'red':   ((0.0,  1, 1),
                   (0.5,  r, r),
                   (1.0,  0, 0)),
         'green': ((0.0,  1, 1),
                   (0.5,  g, g),
                   (1.0,  0, 0)),
         'blue':  ((0.0,  1, 1),
                   (0.5,  b, b),
                   (1.0,  0, 0))}

ACCENT_COLOR_1_CMAP = mpl.colors.LinearSegmentedColormap('testcmap', cdict)
plt.register_cmap(cmap=ACCENT_COLOR_1_CMAP)


def single_fig(figsize=SINGLE_FIG_SIZE):
    return plt.subplots(1, 1, figsize=figsize)


def plot_confusion_matrix(M, labels, ax, cmap=plt.cm.Blues, rng=None):
    """ Plot a confusion matrix on supplied axes.

        Inputs:
        M - (KxK) array-like confusion/mixing matrix
        labels - K-dim vector of string labels
        ax - matplotlib axes object to be drawn upon
        cmap - (optional) mpl-compatible colormap
        Returns:
        matplotlib pcolor results
        Notes:
        Add colorbar with...

        >>> # M, labels both previous defined...
        >>> fig, ax = plt.subplots()
        >>> cm = plot_confusion_matrix(M, labels, ax)
        >>> fig.colorbar(cm)
    """
    if rng is None:
        min_value = M.min()
        max_value = M.max()
    else:
        min_value, max_value = rng

    # if max_value < 1.0:
    #    max_value = 1.0 # matrix is normalized

    heatmap = ax.pcolor(M, cmap=cmap)
    ax.set_xticks(np.arange(M.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(M.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, minor=False)  # add rotation=int to rotate labels
    ax.set_yticklabels(labels, minor=False)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_label_position('top')
    heatmap.set_clim(vmin=min_value, vmax=max_value)
    print(min_value, max_value)

    return heatmap


def color_bp(bp, color):
    """ Helper function for making prettier boxplots """
    c = np.array(color)  # * 0.5
    c = tuple(c)

    for x in bp['boxes']:
        plt.setp(x, color=c)
        x.set_facecolor(color)
    for x in bp['medians']:
        plt.setp(x, color='w')
    for x in bp['whiskers']:
        plt.setp(x, color=c)
    for x in bp['fliers']:
        plt.setp(x, color=c)
    for x in bp['caps']:
        plt.setp(x, color=c)


def adjust_spines(ax, spines):
    """ From http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def hide_right_top_axis(ax):
    """ Remove the top and right axis """
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def finalize(ax, fontsize=LABEL_SIZE, labelpad=7, ignore_legend=False):
    if not ignore_legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) == 0 or len(labels) == 0:
            ignore_legend = True

    """ Apply final adjustments """
    ax.tick_params(direction='out')
    hide_right_top_axis(ax)
    ax.yaxis.label.set_size(fontsize)
    ax.xaxis.label.set_size(fontsize)
    if not ignore_legend:
        ax.legend(frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=labelpad)


def lineswap_axis(fig, ax, zorder=-1000, lw=1, alpha=0.2, skip_zero=False):
    """ Replace y-axis ticks with horizontal lines running through the background.
        Sometimes this looks really cool. Worth having in the bag 'o tricks.
    """
    # fig.canvas.draw()  # Populate the tick vals/labels. Required for get_[xy]ticklabel calls.
    #
    # ylabels = [str(t.get_text()) for t in ax.get_yticklabels()]
    # yticks = [t for t in ax.get_yticks()]
    # xlabels = [str(t.get_text()) for t in ax.get_xticklabels()]
    # xticks = [t for t in ax.get_xticks()]
    #
    # x_draw = [tick for label, tick in zip(ylabels, yticks) if label != '']  # Which ones are real, though?
    # y_draw = [tick for label, tick in zip(ylabels, yticks) if label != '']
    #
    # xmin = x_draw[0]
    # xmax = x_draw[-1]
    #
    # # Draw all the lines
    # for val in y_draw:
    #     if val == 0 and skip_zero:
    #         continue  # Don't draw over the bottom axis
    #     ax.plot([xmin, xmax], [val, val], color=ALMOST_BLACK, zorder=zorder, lw=lw, alpha=alpha)
    #
    # ax.spines["left"].set_visible(False)  # Remove the spine
    # ax.tick_params(axis=u'y', which=u'both', length=0)  # Erase ticks by setting length=0
    # ax.set_xlim(xmin, xmax)  # Retain original xlims
    ax.tick_params(axis=u'y', which=u'both', length=0)
    [ax.spines[axis].set_visible(False) for axis in ["left", "right", "top"]]
    ax.grid(axis='y')

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])