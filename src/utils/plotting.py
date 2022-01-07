import numpy as np
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from simulation.sources import Source


def latex(fraction: Fraction) -> str:
    """
    Convert a fraction.Fraction, in multiples of pi, to a nice
    latex representation
    """
    fraction = str(fraction)
    if '/' in fraction:
        l = fraction.split('/')
        l[0] = l[0].replace('1', '')
        if '-' in l[0]:
            fraction = '-\\' + 'frac{' + l[0][1:] + '\pi}{' + l[1] + '}'
        else:
            fraction = '\\' + 'frac{' + l[0] + '\pi}{' + l[1] + '}'
    elif fraction == '1':
        fraction = '\pi'
    elif fraction == '-1':
        fraction = '-\pi'
    elif fraction == '0':
        pass
    else:
        fraction += '\pi'
    return '${}$'.format(fraction)


def get_pi_ticks(between=(-np.pi, np.pi), step=np.pi / 4):
    """
    Get the positions and labels (string, latexed) for an axis
    in multiples of pi
    """
    start, stop = between
    ticks = np.array(list(np.arange(start, stop, step)) + [stop])
    labels = [latex(Fraction(number).limit_denominator()) for number in ticks / np.pi]
    return ticks, labels


def add_pi_ticks(ax, between=(-np.pi, np.pi), step=np.pi / 4, axis='x'):
    ticks, labels = get_pi_ticks(between=between, step=step)
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == 'y':
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    elif axis == 'both':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def plot_wpb_dist(params: np.array,
                  title: str=None,
                  add_kde: bool=False,
                  y_max: float=None,
                  save_as: str=None,
                  ax=None,
                  legend=True):

    cols = {'w': '#1f77b4', 'p': '#ff7f0e', 'b': '#2ca02c'}

    if ax is None:
        fig, ax = plt.subplots()

    stds = np.std(params, axis=0)
    means = np.mean(params, axis=0)

    for i, typ in enumerate(['w', 'p', 'b']):
        ax.hist(params[:, i],
                 label='${}$ = {:.2f} $\pm$ {:.2f}'.format(typ, means[i], stds[i]),
                 bins=100,
                 alpha=0.6,
                 density=True,
                 color=cols[typ])

    if add_kde:
        y_w = gaussian_kde(params[:, 0])
        y_p = gaussian_kde(params[:, 1])
        y_b = gaussian_kde(params[:, 2])
        x = np.linspace(0, 1, 250)
        ax.plot(x, y_w, color=cols['w'])
        ax.plot(x, y_p, color=cols['p'])
        ax.plot(x, y_b, color=cols['b'])

    if legend:
        plt.legend()

    plt.xlim(0, 1)

    if y_max is not None:
        plt.ylim(0, 40)

    plt.title(title)

    if save_as is not None:
        if '.' in save_as:
            plt.savefig(save_as[:save_as.index('.')] + '.pdf')
        else:
            plt.savefig(save_as + '.pdf')

    plt.show()


def plot_paths(paths: pd.DataFrame, source: Source, ax=None):

    max_x, min_x = paths['x'].max(), paths['x'].min()
    max_y, min_y = paths['y'].max(), paths['y'].min()
    x_diff = max_x - min_x
    y_diff = max_y - min_y

    max_x += 0.05 * x_diff
    min_x -= 0.05 * x_diff
    max_y += 0.05 * y_diff
    min_y -= 0.05 * y_diff

    step = (max_x - min_x) / 500

    x_space = np.arange(min_x, max_x, step)
    y_space = np.arange(min_y, max_y, step)
    X, Y = np.meshgrid(x_space, y_space)

    Rs = (X - source.position[0]) ** 2 + (Y - source.position[1]) ** 2
    Z = np.exp(- Rs)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(Z, extent=[min_x, max_x, min_y, max_y], origin='lower')
    ax.set_aspect('equal')

    for id, path in paths.groupby('trackID'):
        ax.plot(path['x'], path['y'], linewidth=0.5)

    if ax is None:
        plt.show()


def plot_AD_param_dist(dist: np.ndarray, priors: list=None):
    """
    Plot the output distibution of the attractant dynamics parameters

    Parameters
    ----------
    dist        The output MCMC trails (from AttractantInferer().infer())
    priors      A list of the prior distributions (from AttractantInferer().priors)

    """

    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(12, 5), sharex='col')
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    names = ['$q$ [mol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]', '$R_0$ [mol $\mu m^{-2}$]',
             '$\kappa_d$ [mol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mol$^{-1}$]', '$b_0$ [unitless]']

    for j in range(7):
        axes[j].set_title(names[j])
        axes[j].set_yticks([])
        axes[j].hist(dist[:, j], bins=50, color=cols[j], alpha=0.6, density=True)
        if priors is not None:
            priors[j].plot(ax=axes[j], color=cols[j])

    plt.tight_layout()
    plt.show()
