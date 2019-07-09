import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import re
from scipy.stats import norm

class Distribution:

    def __init__(self,
                 mu: Union[float, np.ndarray]=None,
                 sig: Union[float, np.ndarray]=None):

        self.mu = mu
        self.sig = sig

    def pdf(self, x: Union[float, np.ndarray]):
        raise NotImplementedError

    def logpdf(self, x: Union[float, np.ndarray]):
        raise NotImplementedError

    def sample(self, size: Union[tuple, float]=None):
        raise NotImplementedError

    def set_params(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        name = ''.join([' ' + w for w in re.findall('[A-Z][^A-Z]*', self.__class__.__name__)])
        plt.title('Set parameters for{} distribution'.format(name.lower()))

        # Adjust the subplots region to leave some space for the sliders and buttons
        fig.subplots_adjust(bottom=0.25)
        a, b = self.get_xlims()
        self.x = np.linspace(a, b, 1001)
        self.y = self.pdf(self.x)
        [line] = ax.plot(self.x, self.y)

        mu1, mu2 = self.get_mean_lims()
        sig1, sig2 = self.get_sig_lims()


        slider_ax1 = fig.add_axes([0.15, 0.1, 0.65, 0.03])
        slider_ax2 = fig.add_axes([0.15, 0.15, 0.65, 0.03])
        slider1 = Slider(slider_ax1, r'$\mu$', mu1, mu2, valinit=self.mu)
        slider2 = Slider(slider_ax2, r'$\sigma$', sig1, sig2, valinit=self.sig)

        def slider1_on_changed(val):
            self.mu = slider1.val
            self.y = self.pdf(self.x)
            line.set_ydata(self.y)
            fig.canvas.draw_idle()

        def slider2_on_changed(val):
            self.sig = slider2.val
            line.set_ydata(self.pdf(self.x))
            fig.canvas.draw_idle()

        def reset_axis(mouse_event):

            max_y = np.max(self.y)
            min_y = np.min(self.y)
            diff = max_y - min_y
            ax.set_ylim([min_y, max_y + 0.1 * diff])


        slider1.on_changed(slider1_on_changed)
        slider2.on_changed(slider2_on_changed)
        reset_button_ax = fig.add_axes([0.4, 0.025, 0.15, 0.04])
        reset_axis_button = Button(reset_button_ax, 'Reset axis', hovercolor='0.975')
        reset_axis_button.on_clicked(reset_axis)

        plt.show()

    def get_xlims(self):
        raise NotImplementedError

    def get_mean_lims(self):
        raise NotImplementedError

    def get_sig_lims(self):
        raise NotImplementedError

    def plot(self, N=1001):
        a, b,  = self.get_xlims()
        x = np.linspace(a, b, N)
        plt.plot(x, self.pdf(x))

    def cdf(self, x: Union[float, np.ndarray]):
        raise NotImplementedError

    def ppf(self, x: Union[float, np.ndarray]):
        raise NotImplementedError

    def pdf_at_prob(self, x: Union[float, np.ndarray]):
        raise NotImplementedError


class WrappedNormal(Distribution):

    def __init__(self,
                 mu: Union[float, np.ndarray],
                 sig: float):

        super().__init__(mu, sig)
        self.mean = mu
        self.variance = 1 - np.exp(- 0.5 * self.sig ** 2)

    def pdf(self, x: Union[float, np.ndarray]):

        # if sigma is greater than 6, the difference between it and a uniform distribution is ~1e-8
        if self.sig > 4:
            if isinstance(x, np.ndarray):
                return np.ones_like(x) / (2 * np.pi)
            else:
                return 1 / (2 * np.pi)
        if self.sig == 0:
            return self.mu

        # *blindly* opt for 5 loops either side

        # mu =  [ mu1  ,  mu2  ,  mu3  ,  mu4  ]

        #          .        .        .        .
        #          .        .        .        .
        # X =   [a - 2π, b - 2π, c - 2π, d - 2π]
        #       [  a   ,   b   ,   c   ,   d   ]
        #       [a + 2π, b + 2π, c + 2π, d + 2π]
        #       [a + 4π, b + 4π, c + 4π, d + 4π]
        #          .        .        .        .
        #          .        .        .        .

        # then sum normal(X) vertically

        X = np.array([x + 2 * np.pi * i for i in range(-4, 5)])
        return Normal(self.mu, self.sig).pdf(X).sum(0)

    def sample(self, size: Union[tuple, float]=None):
        """
        Return an array of independant draws from a wrapped normal distribution
        """
        return np.mod(np.random.normal(np.pi + self.mu, self.sig, size), 2 * np.pi) - np.pi

    def logpdf(self, x: Union[float, np.ndarray]):
        return np.log(self.pdf(x))

    def get_xlims(self):
        return -np.pi, np.pi

    def get_mean_lims(self):
        return -np.pi, np.pi

    def get_sig_lims(self):
        return 0.01, 6


class Bernoulli(Distribution):

    def __init__(self, mu: float):
        super().__init__(mu=mu)
        self.mean = mu
        self.variance = mu * (1 - mu)

    def sample(self, size: Union[tuple, float]=None):
        return np.random.uniform(0, 1, size) < self.mu


class TruncatedNormal(Distribution):

    def __init__(self,
                 mu: Union[float, np.ndarray],
                 sig: Union[float, np.ndarray]):
        super().__init__(mu=mu, sig=sig)

    def sample(self, size: Union[tuple, float]=None):
        return np.abs(np.random.normal(0, self.sig, size)) + self.mu

    def pdf(self, x: Union[float, np.ndarray]):
        out = (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))
        out[x < self.mu] = 0
        return 2 * out

    def logpdf(self, x: Union[float, np.ndarray]):
        out = - (x - self.mu) ** 2 / (2 * self.sig ** 2) - np.log(self.sig * (2 * np.pi) ** 0.5) + np.log(2)
        out[x < self.mu] = -np.inf
        return out

    def get_xlims(self):
        return self.mu, 5 * self.sig

    def pdf_at_prob(self, x: Union[float, np.ndarray]):
        return 2 * norm(0, self.sig).pdf(norm(0, self.sig).ppf(x / 2))


class Normal(Distribution):

    def __init__(self,
                 mu: Union[float, np.ndarray],
                 sig: Union[float, np.ndarray]):
        super().__init__(mu=mu, sig=sig)

    def sample(self, size: Union[tuple, float]=None):
        return np.random.normal(self.mu, self.sig, size)

    def pdf(self, x: Union[float, np.ndarray]):
        return (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))

    def logpdf(self, x: Union[float, np.ndarray]):
        return - (x - self.mu) ** 2 / (2 * self.sig ** 2) - np.log(self.sig * (2 * np.pi) ** 0.5)

    def get_xlims(self):
        return self.mu - 5 * self.sig, self.mu + 5 * self.sig

    def get_mean_lims(self):

        if self.mu != 0:
            return - 5 * self.mu, 5 * self.mu
        else:
            return -5, 5

    def get_sig_lims(self):
        return 0.01 * self.sig, 10 * self.sig


class LogNormal(Distribution):

    def __init__(self,
                 mu: Union[float, np.ndarray],
                 sig: Union[float, np.ndarray]):
        super().__init__(mu=mu, sig=sig)

    def sample(self, size: Union[tuple, float]=None):
        return np.random.lognormal(self.mu, self.sig, size)

    def pdf(self, x: Union[float, np.ndarray]):
        return (1 / (x * self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (np.log(x) - self.mu) ** 2 / (2 * self.sig ** 2))

    def logpdf(self, x: Union[float, np.ndarray]):
        return - np.log(x * self.sig * (2 * np.pi) ** 0.5) - (np.log(x) - self.mu) ** 2 / (2 * self.sig ** 2)

    def get_xlims(self):
        return np.exp(self.mu - 5 * self.sig), np.exp(self.mu + 5 * self.sig)

    def get_mean_lims(self):
        return -5, 5

    def get_sig_lims(self):
        return 0.01, 10


class Exponential(Distribution):

    def __init__(self,
                 mu: Union[float, np.ndarray]):
        super().__init__(mu=mu)
        self.lamda = 1 / mu
        self.log_lamda = np.log(self.lamda)
        self.sig = mu

    def sample(self, size: Union[tuple, float]=None):
        return np.random.lognormal(self.mu, self.sig, size)

    def pdf(self, x: Union[float, np.ndarray]):
        if isinstance(x, np.ndarray):
            out = np.exp(-x / self.mu) / self.mu
            out[x == 0] = 0
            return out
        if x < 0:
            return 0
        return np.exp(-x / self.mu) / self.mu

    def logpdf(self, x: Union[float, np.ndarray]):
        if isinstance(x, np.ndarray):
            out = self.log_lamda - x * self.lamda
            out[x == 0] = -np.inf
            return out
        if x < 0:
            return -np.inf
        return self.log_lamda - x * self.lamda

    def get_xlims(self):
        return 0, 5 * self.mu

    def get_mean_lims(self):
        return 0, 10

    def get_sig_lims(self):
        return 0.01, 10


if __name__ == '__main__':
    d = Exponential(0.05)
    d.set_params()
