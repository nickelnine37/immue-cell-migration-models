import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, truncnorm, uniform

import scipy

"""
This file contains a variety of distribution classes. They are based roughly
on the format of scipy frozen distributions. They should all, at a minimum,
have methods 'pdf', 'logpdf' and 'sample'. Except for Bernoulli, which is
a discrete distribution with no pdf.

Generally, scipy distributions have accesss to the following methods

rvs(size)       Random variates.
pdf(x)          Probability density function.
logpdf(x)       Log of the probability density function.
cdf(x)          Cumulative distribution function.
logcdf(x)       Log of the cumulative distribution function.
sf(x)           Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
logsf(x)        Log of the survival function.
ppf(q)          Percent point function (inverse of cdf — percentiles).
isf(q)          Inverse survival function (inverse of sf).
moment(n)       Non-central moment of order n
stats()         Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
entropy()       (Differential) entropy of the RV.
fit(data)       Parameter estimates for generic data.
expect(func)    Expected value of a function (of one argument) with respect to the distribution.
median()        Median of the distribution.
mean()          Mean of the distribution.
var()           Variance of the distribution.
std()           Standard deviation of the distribution.
interval(alpha) Endpoints of the range that contains alpha percent of the distribution

 """


class ScipyDistribution(scipy.stats._distn_infrastructure.rv_frozen):
    """
    This is a class that some didstributions can inherit from, which
    gives access to the scipy functionality. The reason for the
    inheritacne is to give them better names, and add some custom
    functionality where necessary.
    """

    def __init__(self, dist_type, *args, **kwargs):
        super().__init__(dist_type, *args, **kwargs)

        self.valid_prior = True


    def sample(self, size: Union[tuple, float]=None):
        return self.rvs(size)

    def plot(self, ax=None, N=1001):

        a, b = self.get_xlims()
        x = np.linspace(a, b, N)
        if ax is None:
            plt.plot(x, self.pdf(x))
        else:
            ax.plot(x, self.pdf(x))

    def get_xlims(self):
        raise NotImplementedError


class LogNormal(ScipyDistribution):

    def __init__(self, mu: Union[float, np.ndarray], sig: Union[float, np.ndarray]):
        """

        A log-normal distribution, where exp(X) is normally distributed.

        Parameters
        ----------
        mu      The mean of the underlying normal distribution.
        sig     The scale of the underlying normal distribution.
        """

        super().__init__(dist_type=lognorm, s=sig, scale=np.exp(mu))

        self.sig = sig
        self.mu = mu


class Normal(ScipyDistribution):

    def __init__(self, mu: Union[float, np.ndarray], sig: Union[float, np.ndarray]):
        """

        A normal distribution.

        Parameters
        ----------
        mu      The mean of the underlying normal distribution.
        sig     The scale of the underlying normal distribution.
        """

        super().__init__(dist_type=norm, loc=mu, scale=sig)

        self.sig = sig
        self.mu = mu

    def get_xlims(self):
        return self.mu - 4 * self.sig, self.mu + 4 * self.sig


class Exponential(ScipyDistribution):

    def __init__(self, lamda: Union[float, np.ndarray]):
        """

        An exponential distribution.

        Parameters
        ----------
        lamda    the lambda parameter, such that the p(x) = lambda * exp(- x * lambda)
        """
        super().__init__(dist_type=expon, scale=1/lamda)
        self.lamda = lamda

    def get_xlims(self):
        return 0, 5 / self.lamda


class Bernoulli:

    def __init__(self, mu: float):
        self.mu = mu
        self.valid_prior = False

    def sample(self, size: Union[tuple, float]=None):
        return np.random.uniform(0, 1, size) < self.mu


class WrappedNormal:

    def __init__(self, mu: Union[float, np.ndarray], sig: float):

        self.mu = mu
        self.sig = sig
        self.valid_prior = True

    def pdf(self, x: Union[float, np.ndarray]):

        # if sigma is greater than 4, the difference between it and a uniform distribution is ~1e-8
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
        return self.normal_pdf(X).sum(0)

    def normal_pdf(self, x: Union[float, np.ndarray]):
        return (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))

    def sample(self, size: Union[tuple, float]=None):
        return np.mod(np.random.normal(np.pi + self.mu, self.sig, size), 2 * np.pi) - np.pi

    def logpdf(self, x: Union[float, np.ndarray]):
        return np.log(self.pdf(x))

    def get_xlims(self):
        return -np.pi, np.pi


class TruncatedNormal(ScipyDistribution):

    def __init__(self, sig: Union[float, np.ndarray]):
        """

        A truncated normal distribution, allowing only positive
        values.

        Parameters
        ----------
        sig    the scale of the normal distribution, which is then truncated.
        """
        super().__init__(dist_type=truncnorm, a=0, b=np.inf, loc=0, scale=sig)
        self.sig = sig


    def get_xlims(self):
        return 0, 5 * self.sig


class Uniform(ScipyDistribution):

    def __init__(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]):
        """

        A truncated normal distribution, allowing only positive
        values.

        Parameters
        ----------
        sig    the scale of the normal distribution, which is then truncated.
        """
        super().__init__(dist_type=uniform)
        self.a = a
        self.b = b


    def get_xlims(self):
        return self.a, self.b



if __name__ == '__main__':

    # x = np.linspace(-2, 5, 1001)
    dist = TruncatedNormal(sig=3)
    # plt.figure()
    # plt.plot(x, dist.pdf(x))
    # plt.show()
    # print(dist.sample((100, 4)))
    # print(dist.mean())
    print(dist.pdf(dist.ppf(1 - 0.99)))



# import numpy as np
# from typing import Union
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
# import re
# from scipy.stats import norm
#
# class Distribution:
#
#     def __init__(self,
#                  mu: Union[float, np.ndarray]=None,
#                  sig: Union[float, np.ndarray]=None):
#
#         self.mu = mu
#         self.sig = sig
#
#     def pdf(self, x: Union[float, np.ndarray]):
#         raise NotImplementedError
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         raise NotImplementedError
#
#     def sample(self, size: Union[tuple, float]=None):
#         raise NotImplementedError
#
#     def set_params(self):
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         name = ''.join([' ' + w for w in re.findall('[A-Z][^A-Z]*', self.__class__.__name__)])
#         plt.title('Set parameters for{} distribution'.format(name.lower()))
#
#         # Adjust the subplots region to leave some space for the sliders and buttons
#         fig.subplots_adjust(bottom=0.25)
#         a, b = self.get_xlims()
#         self.x = np.linspace(a, b, 1001)
#         self.y = self.pdf(self.x)
#         [line] = ax.plot(self.x, self.y)
#
#         mu1, mu2 = self.get_mean_lims()
#         sig1, sig2 = self.get_sig_lims()
#
#
#         slider_ax1 = fig.add_axes([0.15, 0.1, 0.65, 0.03])
#         slider_ax2 = fig.add_axes([0.15, 0.15, 0.65, 0.03])
#         slider1 = Slider(slider_ax1, r'$\mu$', mu1, mu2, valinit=self.mu)
#         slider2 = Slider(slider_ax2, r'$\sigma$', sig1, sig2, valinit=self.sig)
#
#         def slider1_on_changed(val):
#             self.mu = slider1.val
#             self.y = self.pdf(self.x)
#             line.set_ydata(self.y)
#             fig.canvas.draw_idle()
#
#         def slider2_on_changed(val):
#             self.sig = slider2.val
#             line.set_ydata(self.pdf(self.x))
#             fig.canvas.draw_idle()
#
#         def reset_axis(mouse_event):
#
#             max_y = np.max(self.y)
#             min_y = np.min(self.y)
#             diff = max_y - min_y
#             ax.set_ylim([min_y, max_y + 0.1 * diff])
#
#
#         slider1.on_changed(slider1_on_changed)
#         slider2.on_changed(slider2_on_changed)
#         reset_button_ax = fig.add_axes([0.4, 0.025, 0.15, 0.04])
#         reset_axis_button = Button(reset_button_ax, 'Reset axis', hovercolor='0.975')
#         reset_axis_button.on_clicked(reset_axis)
#
#         plt.show()
#
#     def get_xlims(self):
#         raise NotImplementedError
#
#     def get_mean_lims(self):
#         raise NotImplementedError
#
#     def get_sig_lims(self):
#         raise NotImplementedError
#
#     def plot(self, N=1001):
#         a, b,  = self.get_xlims()
#         x = np.linspace(a, b, N)
#         plt.plot(x, self.pdf(x))
#
#     def cdf(self, x: Union[float, np.ndarray]):
#         raise NotImplementedError
#
#     def ppf(self, x: Union[float, np.ndarray]):
#         raise NotImplementedError
#
#     def pdf_at_prob(self, x: Union[float, np.ndarray]):
#         raise NotImplementedError
#
#
# class WrappedNormal(Distribution):
#
#     def __init__(self,
#                  mu: Union[float, np.ndarray],
#                  sig: float):
#
#         super().__init__(mu, sig)
#         self.mean = mu
#         self.variance = 1 - np.exp(- 0.5 * self.sig ** 2)
#
#     def pdf(self, x: Union[float, np.ndarray]):
#
#         # if sigma is greater than 6, the difference between it and a uniform distribution is ~1e-8
#         if self.sig > 4:
#             if isinstance(x, np.ndarray):
#                 return np.ones_like(x) / (2 * np.pi)
#             else:
#                 return 1 / (2 * np.pi)
#         if self.sig == 0:
#             return self.mu
#
#         # *blindly* opt for 5 loops either side
#
#         # mu =  [ mu1  ,  mu2  ,  mu3  ,  mu4  ]
#
#         #          .        .        .        .
#         #          .        .        .        .
#         # X =   [a - 2π, b - 2π, c - 2π, d - 2π]
#         #       [  a   ,   b   ,   c   ,   d   ]
#         #       [a + 2π, b + 2π, c + 2π, d + 2π]
#         #       [a + 4π, b + 4π, c + 4π, d + 4π]
#         #          .        .        .        .
#         #          .        .        .        .
#
#         # then sum normal(X) vertically
#
#         X = np.array([x + 2 * np.pi * i for i in range(-4, 5)])
#         return Normal(self.mu, self.sig).pdf(X).sum(0)
#
#     def sample(self, size: Union[tuple, float]=None):
#         """
#         Return an array of independant draws from a wrapped normal distribution
#         """
#         return np.mod(np.random.normal(np.pi + self.mu, self.sig, size), 2 * np.pi) - np.pi
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         return np.log(self.pdf(x))
#
#     def get_xlims(self):
#         return -np.pi, np.pi
#
#     def get_mean_lims(self):
#         return -np.pi, np.pi
#
#     def get_sig_lims(self):
#         return 0.01, 6
#
#
# class Bernoulli(Distribution):
#
#     def __init__(self, mu: float):
#         super().__init__(mu=mu)
#         self.mean = mu
#         self.variance = mu * (1 - mu)
#
#     def sample(self, size: Union[tuple, float]=None):
#         return np.random.uniform(0, 1, size) < self.mu
#
#
# class TruncatedNormal(Distribution):
#
#     def __init__(self,
#                  sig: Union[float, np.ndarray]):
#         super().__init__(mu=0, sig=sig)
#
#     def sample(self, size: Union[tuple, float]=None):
#         return np.abs(np.random.normal(0, self.sig, size)) + self.mu
#
#     def pdf(self, x: Union[float, np.ndarray]):
#         out = (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))
#         out[x < self.mu] = 0
#         return 2 * out
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         out = - (x - self.mu) ** 2 / (2 * self.sig ** 2) - np.log(self.sig * (2 * np.pi) ** 0.5) + np.log(2)
#         out[x < self.mu] = -np.inf
#         return out
#
#     def get_xlims(self):
#         return self.mu, 5 * self.sig
#
#     def pdf_at_prob(self, x: Union[float, np.ndarray]):
#         return 2 * norm(0, self.sig).pdf(norm(0, self.sig).ppf(x / 2))
#
#
# class Normal(Distribution):
#
#     def __init__(self,
#                  mu: Union[float, np.ndarray],
#                  sig: Union[float, np.ndarray]):
#         super().__init__(mu=mu, sig=sig)
#
#     def sample(self, size: Union[tuple, float]=None):
#         return np.random.normal(self.mu, self.sig, size)
#
#     def pdf(self, x: Union[float, np.ndarray]):
#         return (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         return - (x - self.mu) ** 2 / (2 * self.sig ** 2) - np.log(self.sig * (2 * np.pi) ** 0.5)
#
#     def get_xlims(self):
#         return self.mu - 5 * self.sig, self.mu + 5 * self.sig
#
#     def get_mean_lims(self):
#
#         if self.mu != 0:
#             return - 5 * self.mu, 5 * self.mu
#         else:
#             return -5, 5
#
#     def get_sig_lims(self):
#         return 0.01 * self.sig, 10 * self.sig
#
#
# class LogNormal(Distribution):
#
#     def __init__(self,
#                  mu: Union[float, np.ndarray],
#                  sig: Union[float, np.ndarray]):
#         super().__init__(mu=mu, sig=sig)
#
#     def sample(self, size: Union[tuple, float]=None):
#         return np.random.lognormal(self.mu, self.sig, size)
#
#     def pdf(self, x: Union[float, np.ndarray]):
#         return (1 / (x * self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (np.log(x) - self.mu) ** 2 / (2 * self.sig ** 2))
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         return - np.log(x * self.sig * (2 * np.pi) ** 0.5) - (np.log(x) - self.mu) ** 2 / (2 * self.sig ** 2)
#
#     def get_xlims(self):
#         return np.exp(self.mu - 5 * self.sig), np.exp(self.mu + 5 * self.sig)
#
#     def get_mean_lims(self):
#         return -5, 5
#
#     def get_sig_lims(self):
#         return 0.01, 10
#
#
# class Exponential(Distribution):
#
#     def __init__(self,
#                  mu: Union[float, np.ndarray]):
#         super().__init__(mu=mu)
#         self.lamda = 1 / mu
#         self.log_lamda = np.log(self.lamda)
#         self.sig = mu
#
#     def sample(self, size: Union[tuple, float]=None):
#         return np.random.lognormal(self.mu, self.sig, size)
#
#     def pdf(self, x: Union[float, np.ndarray]):
#         if isinstance(x, np.ndarray):
#             out = np.exp(-x / self.mu) / self.mu
#             out[x == 0] = 0
#             return out
#         if x < 0:
#             return 0
#         return np.exp(-x / self.mu) / self.mu
#
#     def logpdf(self, x: Union[float, np.ndarray]):
#         if isinstance(x, np.ndarray):
#             out = self.log_lamda - x * self.lamda
#             out[x == 0] = -np.inf
#             return out
#         if x < 0:
#             return -np.inf
#         return self.log_lamda - x * self.lamda
#
#     def get_xlims(self):
#         return 0, 5 * self.mu
#
#     def get_mean_lims(self):
#         return 0, 10
#
#     def get_sig_lims(self):
#         return 0.01, 10