import numpy as np
from fractions import Fraction

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