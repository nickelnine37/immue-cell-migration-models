import numpy as np
from typing import Union
import scipy
from scipy.stats import norm, lognorm, expon, truncnorm, uniform
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])

class WrappedNormal:

    def __init__(self, mu: Union[float, np.ndarray], sig: float):

        self.mu = self.domain_map(mu)
        self.sig = sig
        self.valid_prior = True

        # This slightly cryptic line tells us how many times we should wrap the distribution
        # in each direection before giving up. This will gaurantee negligible error, whilst
        # not doing more loops than necessary. Tested against worst case scenario of mu = +/-pi
        self.n_wraps = int(np.ceil(1.3 * sig))

    def pdf(self, theta: Union[float, np.ndarray]):

        # if sigma is greater than 4, the difference between it and a uniform distribution is ~1e-8
        if self.sig > 4:
            if isinstance(theta, np.ndarray):
                return np.ones_like(theta) / (2 * np.pi)
            else:
                return 1 / (2 * np.pi)

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

        theta = self.domain_map(theta)
        X = np.array([theta + i * 2 * np.pi for i in range(-self.n_wraps, self.n_wraps+1)])
        return self.normal_pdf(X).sum(0)

    def normal_pdf(self, x: Union[float, np.ndarray]):
        return (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))

    def sample(self, size: Union[tuple, float]=None):
        return self.domain_map(np.random.normal(self.mu, self.sig, size))

    def logpdf(self, theta: Union[float, np.ndarray]):
        return np.log(self.pdf(theta))

    def domain_map(self, theta: Union[float, np.ndarray]):
        return np.mod(theta + np.pi, 2 * np.pi) - np.pi


class Bernoulli:

    def __init__(self, mu: float):
        self.mu = mu
        self.valid_prior = False

    def sample(self, size: Union[tuple, float] = None):
        return np.random.uniform(0, 1, size) < self.mu


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

    def plot(self, ax=None, N=1001, color=None):

        a, b = self.get_xlims()
        x = np.linspace(a, b, N)
        if ax is None:
            plt.plot(x, self.pdf(x), color=color)
        else:
            ax.plot(x, self.pdf(x), color=color)

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

