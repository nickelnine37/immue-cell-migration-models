import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
from scipy.stats import multivariate_normal
from utils.distributions import WrappedNormal, Uniform, Normal
from inference.base_inference import inferer
from typing import Union
from utils.exceptions import SquareRootError
from in_silico.sources import Wound, PointWound, CellsInsideWound, CellsOnWoundMargin

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
    k = (0.25 * (kappa + R_0 + A) ** 2 - R_0 * A)

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


class AttractantInferer(inferer):

    def __init__(self, ob_readings: dict, wound: Wound, priors: list=None, t_units='minutes'):
        """
        Perform inference on observed bias readings to infer the posterior distribution over the
        attractant dynamics parameters {q, D, τ, R0, κ, m, b0}.

        A dictionary specifying the observed bias readings must be provided, along with a certain
        instantiated wound (which can be a PointWound, a CellsOnWoundMargin or CellsInsideWound) .

        The observed bias readings should be a dictionary with elements of the following form:

        {(r1, t1): (mu1, sig1), (r2, t2): (mu2, sig2) ... }

        r and t specify the spatial and temporal location where the observed bias has been measured,
        (this could be the mid-point of their respective bins), and mu and sig represent the mean and
        standard deviation of the posterior of the observed bias at this location.

        DISTANCES SHOULD BE MEASURED IN MICRONS

        time can be measured in minutes or seconds: specify this with the t_units argument.

        The parameters are measured in the following units:

        q:      Mmol / min
        D:      µm^2 / min
        τ:      min
        R0:     Mmol / µm^2
        kappa:  Mmol / µm^2
        m:      µm^2 / Mmol
        b0:     unitless


        Parameters
        ----------
        ob_readings     The observed bias readings
        wound           A Wound class, which the observed bias is assumed to be generated from
        priors          A list of distributions, one element per parameter, specifying the priors
        t_units         The units which time is measured in, in the ob_readings dictionary keys
        """

        super().__init__()

        self.wound = wound

        assert t_units in ['seconds', 'minutes'], 't_units must be either "seconds" or "minutes" but it is {}'.format(t_units)

        # the total number of readings
        self.TS = len(ob_readings)

        # extract a list of rs, ts, mus and sigs
        self.r = np.array([r for r, t in ob_readings.keys()])
        self.t = np.array([t for r, t in ob_readings.keys()])
        mus = np.array([mu for mu, sig in ob_readings.values()])
        sigs = np.array([sig for mu, sig in ob_readings.values()])

        # convert to minutes
        if t_units is 'seconds':
            self.t /= 60

        # this is our multivariate Gaussian observed bias distribution
        self.ob_dists = multivariate_normal(mus, sigs ** 2)

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
            return self.ob_dists.logpdf(observed_bias(params, self.r, self.t, self.wound))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: np.ndarray):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])


if __name__ == '__main__':

    from utils.plotting import plot_AD_param_dist

    # TEST

    # here are some example observed bias readings
    ob_readings = {(25, 10): (0.1732, 0.02),
                   (50, 10): (0.1541, 0.02),
                   (75, 10): (0.1081, 0.02),
                   (100, 10): (0.0647, 0.02),
                   (125, 10): (0.0349, 0.02),
                   (150, 10): (0.0174, 0.02),
                   (25, 30): (0.1018, 0.02),
                   (50, 30): (0.1007, 0.02),
                   (75, 30): (0.0955, 0.02),
                   (100, 30): (0.082, 0.02),
                   (125, 30): (0.0659, 0.02),
                   (150, 30): (0.0500, 0.02),
                   (25, 50): (0.0077, 0.02),
                   (50, 50): (0.0141, 0.02),
                   (75, 50): (0.0196, 0.02),
                   (100, 50): (0.0238, 0.02),
                   (125, 50): (0.0263, 0.02),
                   (150, 50): (0.0271, 0.02),
                   (25, 80): (0.00309, 0.02),
                   (50, 80): (0.00509, 0.02),
                   (75, 80): (0.00693, 0.02),
                   (100, 80): (0.0085, 0.02),
                   (125, 80): (0.0098, 0.02),
                   (150, 80): (0.0107, 0.02),
                   (25, 120): (0.0018, 0.02),
                   (50, 120): (0.0026, 0.02),
                   (75, 120): (0.0034, 0.02),
                   (100, 120): (0.004, 0.02),
                   (125, 120): (0.004, 0.02),
                   (150, 120): (0.005, 0.02)}

    # make a new inferer and infer the distribution over underlying parameters
    inferer = AttractantInferer(ob_readings, PointWound())
    dist_out = inferer.multi_infer(n_walkers=5,
                                   n_steps=300000,
                                   burn_in=100000,
                                   suppress_warnings=True)

    # plot the distribution
    plot_AD_param_dist(dist_out, priors=inferer.priors)
