import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from subprocess import check_output
import os
import time

def make_gif(array: np.ndarray, save_as: str, delay: int=10,
             time_first: bool=True, v_bounds: tuple=(None, None),
             dpi: int=None, cells: list=None, paths: list=None,
             extent: list=None, origin: str=None):
    """

    Make a gif file from a numpy array, using Matplotlib imshow over the x-y
    coordinates. Useful for plotting cell tracks and heat equation plots.
    Relies on the linux command line tool 'convert'.

    Parameters
    ----------
    array         A 3D numpy array of shape (T, X, Y) or (X, Y, T).
    save_as       String file name or path to save gif to. No file extension.
    delay         Time delay in ms between gif frames
    time_first    Boolean specifying whether time dimension appears first
    v_bounds      tuple of (vmin, vmax) to be passed to imshow.
    dpi           Resolution in dots per inch

        If working with pictures of cells:

    add_LOG       Whether to add a Laplacian of Gaussian circles to identify cells
    threshold     The threshold to use with LOG
    add_paths     Whether to plot particle paths

    """

    # check if the t-dimension is in index 0
    if not time_first:
        array = array.transpose((2, 0, 1))

    # useful for keeping heat equation time frames consistent
    vmin, vmax = v_bounds
    T, by, bx = array.shape

    # use a system of binary sequences to name the files (helps keep them in order ;P)
    bits = int(np.ceil(np.log2(T)))

    path_to_gif = '/'.join(save_as.split('/')[:-1]) + '/'
    if path_to_gif == '/':
        path_to_gif = './'

    if not os.path.exists(path_to_gif + 'tmp'):
        os.mkdir(path_to_gif + 'tmp')

    path_to_png = path_to_gif + 'tmp/'

    T0 = time.time()
    print('Plotting images')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if paths is not None:
        paths = [(t0, colors[i%len(colors)], path) for i, (t0, path) in enumerate(paths)]

    for t in range(T):
        # sort name stuff
        name = bin(t)[2:]
        name = '0' * (bits - len(name)) + name
        # plot the image
        image = array[t, :, :]
        fig, ax = plt.subplots()
        plt.imshow(image, vmin=vmin, vmax=vmax, extent=extent, origin=origin)
        plt.title('t={:.2f}'.format(t))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, bx])
        ax.set_ylim([0, by])

        # add Laplacian of Gaussians
        if cells is not None:
            for x, y, r in cells[t]:
                c = plt.Circle((x, y), r, color='red', linewidth=0.5, fill=False)
                ax.add_patch(c)

        if paths is not None:
            for t0, color, path in paths:
                n_points = path.shape[0]
                if n_points > 1:
                    if t0 < t:
                        to = min(n_points, t - t0 + 1)
                        plt.plot(path[:to, 0], path[:to, 1], color=color, linewidth=1, alpha=0.75)
            # plt.show()

        # save each png file
        plt.savefig(path_to_png + '{}.png'.format(name), dpi=dpi)
        plt.close()

    print('Plotted images in {:.2f}s'.format(time.time() - T0))
    print('Creating gif')

    # convert pngs to gif
    check_output(['convert', '-delay', '{}'.format(delay), path_to_png + '*.png', save_as + '.gif'])

    for file_name in os.listdir(path_to_png):
        if '.png' in file_name:
            os.remove(path_to_png + file_name)

    os.rmdir(path_to_png)

    print('Done')




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
                  save_as: str=None):

    cols = {'w': '#1f77b4', 'p': '#ff7f0e', 'b': '#2ca02c'}

    plt.figure()
    stds = np.std(params, axis=0)
    means = np.mean(params, axis=0)

    for i, typ in enumerate(['w', 'p', 'b']):
        plt.hist(params[:, i],
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
        plt.plot(x, y_w, color=cols['w'])
        plt.plot(x, y_p, color=cols['p'])
        plt.plot(x, y_b, color=cols['b'])

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
