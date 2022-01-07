import sys
sys.path.append('..')

import numpy as np
import pickle
import time
from scipy.stats import multivariate_normal
from utils.distributions import WrappedNormal, Uniform, Normal
from inference.base import Inferer
from typing import Union
from utils.exceptions import SquareRootError
from simulation.sources import Wound, PointWound, CellsInsideWound, CellsOnWoundMargin

# This is the Leukocyte radius: 15µm
dr = 15


def complexes(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float], wound: Wound) -> Union[np.ndarray, float]:
    """
    Given a set of AD parameters, this function returns the concentration of bound complexes at radial
    distance r and time t.

    Parameters
    ----------
    params  A numpy array holding at least [tau, q, D, kappa, R_0] in that order
    r       the radial distance from the origin. Can be float or array
    t       time

    Returns
    -------
    C       The concentration of complexes

    """

    q, D, tau, R_0, kappa = params[:5]
    A = wound.concentration(params, r, t)
    k = 0.25 * (kappa + R_0 + A) ** 2 - R_0 * A

    if np.array(k < 0).any():
        raise SquareRootError('Value to be square-rooted in complex eqn is less than zero')

    return 0.5 * (kappa + R_0 + A) - k ** 0.5


def observed_bias(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float], wound: Wound) -> Union[np.ndarray, float]:
    """
    For a set of parameters, calculate the observed bias that would occur
    at a radial distance r and time t for leukocytes of radius dr.

    Parameters
    ----------
    params      a tuple containing q, D, tau, R_0, kappa, m, b_0
    r           the spatial descriptor - distance from the wound
    t           the temporal descriptor - when is this occuring

    Returns
    -------
    The observed bias

    """

    q, D, tau, R_0, kappa, m, b_0 = params
    return m * (complexes(params, r - dr, t, wound) - complexes(params, r + dr, t, wound)) + b_0


class AttractantInferer(Inferer):

    def __init__(self,
                 ob_mean: np.ndarray,
                 ob_sd: np.ndarray,
                 ob_radii: np.ndarray,
                 ob_time: np.ndarray,
                 wound: Wound,
                 priors: list=None,
                 t_units='minutes'):
        """
        Perform inference on observed bias readings to infer the posterior distribution over the
        attractant dynamics parameters {q, D, τ, R0, κ, m, b0}.

        * DISTANCES SHOULD BE MEASURED IN MICRONS
        * Time can be measured in minutes or seconds: specify this with the t_units argument.
        * The parameters are measured in the following units:

        q:      Mmol / min
        D:      µm^2 / min
        τ:      min
        R0:     Mmol / µm^2
        kappa:  Mmol / µm^2
        m:      µm^2 / Mmol
        b0:     unitless

        Parameters
        ----------
        ob_mean         A numpy array, specifying the mean of the observed bias readings
        ob_sd           A numpy array, specifying the standard deviation of the observed bias readings
        ob_radii        A numpy array, specifying the mean radii that the readings were taken at
        ob_time         A numpy array, specifying the mean time the readings were taken at
        wound           A Wound class, which the observed bias is assumed to be generated from
        priors          A list of distributions, one element per parameter, specifying the priors
        t_units         The units which time is measured in, in the ob_readings dictionary keys
        """

        super().__init__()

        # assertions
        assert t_units in ['seconds', 'minutes'], 't_units must be either "seconds" or "minutes" but it is {}'.format(t_units)
        assert all([isinstance(item, np.ndarray) for item in [ob_mean, ob_radii, ob_sd, ob_time]]), 'first 4 arguments must be numpy arrays'
        assert len(set([len(item) for item in [ob_mean, ob_radii, ob_sd, ob_time]])) == 1, 'all arrays must have the same length'


        self.wound = wound
        self.ob_radii = ob_radii
        self.ob_time = ob_time
        self.ob_mean = ob_mean
        self.ob_sd = ob_sd

        # convert to minutes
        if t_units is 'seconds':
            self.ob_time /= 60

        # this is our multivariate Gaussian observed bias distribution
        self.ob_dist = multivariate_normal(ob_mean, ob_sd ** 2)

        # these are the default priors
        if priors is None:

            self.priors = [Normal(5 * 60, 4 * 60),
                           Normal(400, 300),
                           Normal(60, 16),
                           Normal(0.3, 0.2),
                           Normal(0.1, 0.2),
                           Normal(4, 4),
                           Normal(0.001, 0.0005)]

        else:
            assert isinstance(priors, list)
            assert len(priors) == 7
            self.priors = priors

    def log_likelihood(self, params: np.ndarray):
        """
        For a set of parameters, calculate the log-likelihood of these
        parameters

        Parameters
        ----------
        params      a tuple containing q, D, tau, R_0, kappa, m, b_0

        Returns
        -------
        The log likelihood
        """
        try:
            return self.ob_dist.logpdf(observed_bias(params, self.ob_radii, self.ob_time, self.wound))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: np.ndarray):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])

    def infer(self,
              n_chains: int=5,
              n_steps: int=50000,
              burn_in: int=5000,
              target_ar: float=0.25,
              seed: int=0,
              suppress_warnings: bool=False,
              progress_bar: bool=True) -> np.ndarray:

        return super().infer(n_chains, n_steps, burn_in, target_ar, seed, suppress_warnings, progress_bar)


if __name__ == '__main__':

    from utils.plotting import plot_AD_param_dist
    import matplotlib.pyplot as plt

    # preset some parameters
    q = 600
    D = 125
    tau = 15
    R_0 = 0.3
    kappa = 0.1
    m = 3
    b = 0
    params = np.array([q, D, tau, R_0, kappa, m, b])

    wound = PointWound()

    # the points in space and time where we want to observe the cell bias
    r = np.array([ 25, 50, 75, 100, 125, 150, 25, 50, 75, 100, 125, 150, 25, 50, 75, 100, 125, 150, 25, 50, 75, 100, 125, 150, 25, 50, 75, 100, 125, 150])
    t = np.array([ 10, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 80, 80, 80, 80, 80, 80, 120, 120, 120, 120, 120, 120])

    # create some fake 'observations' by putting pre-set params through model
    bias_mean = observed_bias(params, r, t, wound)
    bias_sd = (bias_mean ** 0.5) / 20

    # make a new inferer and infer the distribution over underlying parameters
    inferer = AttractantInferer(bias_mean,
                                bias_sd,
                                r,
                                t,
                                PointWound(),
                                use_stan=False,
                                recompile_model=True)

    dist_out = inferer.infer(n_chains=5,
                             n_steps=50000,
                             burn_in=5000,
                             suppress_warnings=True,
                             seed=101,
                             target_ar=0.15)


    # plot the posterior marginals
    plot_AD_param_dist(dist_out, priors=inferer.priors)

    # plot the variation of observed bias collected, compared with param posteriors
    r_space = np.linspace(20, 160, 1000)

    T = [10, 30, 50, 80, 120]
    R = [25, 50, 75, 100, 125, 150]

    mean_rs = bias_mean.reshape(5, 6)
    sd_rs = bias_sd.reshape(5, 6)

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True)
    for j, (tt, ax) in enumerate(zip(T, axes)):

        curve = observed_bias(params, r_space, tt, wound)
        ax.text(170, curve.max() / 2, 't = {}'.format(tt))

        ax.plot(r_space, curve, color='#d62728')
        bp = ax.boxplot(x=[(m - 2 * s, m, m + 2 * s) for m, s in zip(mean_rs[j, :], sd_rs[j, :])], positions=R,
                        widths=4)

        for piece in ['boxes', 'whiskers', 'medians', 'caps']:
            for lines in bp[piece]:
                lines.set(linewidth=0.5)

        for i in np.random.randint(0, 100000, size=100):
            ax.plot(r_space, observed_bias(dist_out[i, :], r_space, tt, wound), alpha=0.03, color='#1f77b4')

    ax.set_xlabel('Distance from wound, microns')
    plt.show()



