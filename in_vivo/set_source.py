from matplotlib.patches import Circle
import skimage
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from matplotlib.widgets import Button
import argparse
from in_silico.sources import PointSource
import numpy as np



def set_source(frames):

    fig, ax = plt.subplots()

    T, by, bx = frames.shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, bx])
    ax.set_ylim([0, by])

    class SourceSetter:

        def __init__(self, fig, ax):
            self.fig, self.ax = fig, ax

            self.circ = Circle((None, None), 30, fill=False, linewidth=1, color='red')
            self.ax.add_artist(self.circ)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        def on_click(self, event):
            if event.inaxes == self.ax:
                self.x, self.y = event.xdata, event.ydata
                self.circ.center = self.x, self.y
                self.fig.canvas.draw()

        def run(self):

            self.done = False

            ax.imshow(frames[0, :, :], origin='lower')
            t = 1

            ok_button_ax = fig.add_axes([0.4, 0.025, 0.15, 0.04])
            ok_axis_button = Button(ok_button_ax, 'OK', hovercolor='0.975')
            ok_axis_button.on_clicked(self.OK)

            while True:
                ax.imshow(frames[t % T, :, :], origin='lower')
                plt.pause(0.2)
                del ax.images[0]
                t += 1
                if self.done:
                    return PointSource(position=np.array([self.x, self.y]))

        def OK(self, mouse_event):
            print('Initialise as: PointSource(position=np.array([{:.2f}, {:.2f}]))'.format(self.x, self.y))
            self.done = True
            plt.close()

    ss = SourceSetter(fig, ax)

    return ss.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set the source position')
    parser.add_argument('tif_file', type=str, nargs=1, help='load a tif file')
    parser.add_argument('color_channel', type=int, nargs=1, help='load a tif file', default=1)

    args = parser.parse_args()

    frames = skimage.io.imread(args.tif_file[0])[:, args.color_channel[0], :, :]
    set_source(frames)

