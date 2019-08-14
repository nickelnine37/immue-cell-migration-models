import sys
sys.path.append('..')

import numpy as np
from scipy.stats import multivariate_normal
from utils.distributions import WrappedNormal, Uniform, Normal
from inference.base import MCMC
from typing import Union
from scipy.special import expi
from utils.exceptions import SquareRootError
from in_silico.sources import Wound, PointWound, CellsInsideWound, CellsOnWoundMargin

# This is the Leukocyte radius: 15µm
dr = 15

def complexes(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float], wound: Wound) -> Union[np.ndarray, float]:
    """
    This returns the concentration of bound complexes at radial
    distance r and time t.

    Parameters
    ----------
    r       the radial distance from the origin. Can be float or array
    t       time
    tau     time attractant is released for
    q       flow rate of attractant
    D       diffusion constant
    kappa   diffusivity constant
    R_0     local receptor concentration

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



class AttractantInference(MCMC):

    def __init__(self, ob_readings: dict, wound: Wound, priors: list=None, t_units='minutes'):
        """
        Perform inference on observed bias readings to infer the posterior distribution over the
        attractant dynamics parameters {q, D, τ, R0, kappa, m, b0}.

        A dictionary specifying the observed bias readings must be provided, along with a certain
        instantiated wound (which can be a PointWound, a CellsOnWoundMargin or CellsInsideWound) .

        The observed bias readings should be a dictionary with elements of the following form:

        {(r1, t1): (mu1, sig1), (r2, t2): (mu2, sig2) ... }

        r and t specify the spatial and temporal location where the observed bias has been measured,
        and mu and sig represent the mean and standard deviation of the posterior of the observed
        bias at this location.

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
        self.r = np.array([r for (r, t), (mu, sig) in ob_readings.items()])
        self.t = np.array([t for (r, t), (mu, sig) in ob_readings.items()])
        mus = np.array([mu for (r, t), (mu, sig) in ob_readings.items()])
        sigs = np.array([sig for (r, t), (mu, sig) in ob_readings.items()])

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

    import time
    import matplotlib.pyplot as plt

    bins = [((0,    900),  (0, 25)), ((0,    900),  (25, 50)), ((0,    900),  (50, 100)), ((0,    900),  (100, 150)),
            ((900,  1800), (0, 25)), ((900,  1800), (25, 50)), ((900,  1800), (50, 100)), ((900,  1800), (100, 150)),
            ((1800, 2700), (0, 25)), ((1800, 2700), (25, 50)), ((1800, 2700), (50, 100)), ((1800, 2700), (100, 150)),
            ((2700, 3600), (0, 25)), ((2700, 3600), (25, 50)), ((2700, 3600), (50, 100)), ((2700, 3600), (100, 150)),
            ((3600, 5400), (0, 25)), ((3600, 5400), (25, 50)), ((3600, 5400), (50, 100)), ((3600, 5400), (100, 150)),
            ((5400, 7200), (0, 25)), ((5400, 7200), (25, 50)), ((5400, 7200), (50, 100)), ((5400, 7200), (100, 150))]

    means = np.array([0.437128, 0.391095, 0.212341, 0.103723,
                      0.269382, 0.286244, 0.203868, 0.127678,
                      0.214605, 0.233463, 0.186637, 0.127907,
                      0.185555, 0.201183, 0.171854, 0.124962,
                      0.159903, 0.170442, 0.154173, 0.119456,
                      0.003758, 0.008397, 0.015033, 0.022781])

    stds = np.array([0.04221,  0.013292, 0.00804,  0.019632,
                     0.019825, 0.01067,  0.009917, 0.03512,
                     0.049061, 0.010166, 0.011422, 0.064734,
                     0.010487, 0.010171, 0.012229, 0.069327,
                     0.027498, 0.033695, 0.011105, 0.030786,
                     0.037766, 0.027568, 0.025019, 0.050603])

    wound = PointWound()

    wbs = np.random.normal(loc=means, scale=stds, size=(20000, 24))
    AD = AttractantInference(wbs, bins, wound)


    init_params = np.array([AD.priors[i].mu for i in range(7)])

    t0 = time.time()
    ps = AD.infer(init_params, n_steps=500000, burn_in=500000, suppress_warnings=True)
    t1 = time.time()

    print('Inference completed in {:.2f}s'.format(t1 - t0))

    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes = axes.reshape(-1)

    for i, name in enumerate(['q', 'D', 'tau', 'R_0', 'kappa', 'm', 'b_0']):
        axes[i].hist(ps[:, i], bins=100, density=True)
        AD.priors[i].plot(axes[i])
        axes[i].set_title(name)
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()

