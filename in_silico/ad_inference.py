import numpy as np
import time
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Union, Iterable
from scipy.special import expi

from utils.distributions import LogNormal, Normal, Exponential
from utils.exceptions import SquareRootError

# np.set_printoptions(linewidth=1000, precision=4, suppress=True)

# This is the Leukocyte radius: 15µm
dr = 15


def concentration(params: Iterable, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
    """
    This function returns the concentration of attractant at a
    radial distance r and time t for a continuous point source
    emitting attractand at a rate q from the origin from t=0
    to t=tau with diffusion constant D.

    Parameters
    ----------
    r       the radial distance from the origin. Can be float or array
    t       time
    tau     time attractant is released for
    q       flow rate of attractant
    D       diffusion constant

    Returns
    -------
    A       The attractant concentration

    """

    q, D, tau = params[:3]

    if isinstance(r, (float, int, np.ndarray)) and isinstance(t, (float, int)):

        if t < tau:
            return - (q / (4 * np.pi * D)) * expi(- r ** 2 / (4 * D * t))
        else:
            return (q / (4 * np.pi * D)) * (expi(- r ** 2 / (4 * D * (t - tau))) - expi(- r ** 2 / (4 * D * t)))

    elif isinstance(r, (float, int, np.ndarray)) and isinstance(t, np.ndarray):

        assert r.shape == t.shape, 'r and t must be the same shape, but they have shapes {} and {} respectively'.format(
            r.shape, t.shape)

        out = - expi(- r ** 2 / (4 * D * t))
        out[t > tau] += expi(- r[t > tau] ** 2 / (4 * D * (t[t > tau] - tau)))

        return (q / (4 * np.pi * D)) * out

    else:
        raise TypeError('r and t must both be floats/ints or numpy arrays')


def complexes(params: Iterable, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
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
    A = concentration(params, r, t)
    k = (0.25 * (kappa + R_0 + A) ** 2 - R_0 * A)

    if np.array(k < 0).any():
        raise SquareRootError('Value to be square-rooted in complex eqn is less than zero')

    return 0.5 * (kappa + R_0 + A) - k ** 0.5


def observed_bias(params: Iterable, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
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
    return m * (complexes(params, r - dr, t) - complexes(params, r + dr, t)) + b_0


class AttractantDynamics:

    def __init__(self, wb: np.ndarray, ts: list, dist_type='kde') -> None:
        """
        Given a numpy array containing the observed bias readings, intialise
        an attractant dynamics inference class. The priors are also set here.

        Parameters
        ----------
        wb      shape: (N, TS). The observed bias readings for T temporal and S spatial clusters.
        ts      a list of (time, distance) readings that define the temporal/spatial clusters
        """

        assert len(ts) == wb.shape[1], 'Must provide same number of (t, r) readings as observed bias readings'


        self.wb = wb
        self.ts = np.array(ts)
        self.TS = len(ts)
        self.r = self.ts[:, 1]
        self.t = self.ts[:, 0]

        if dist_type in ['gaussian', 'Gaussian']:
            self.ob_dists = multivariate_normal(wb.mean(0), wb.var(0))
        elif dist_type in ['kde', 'KDE', 'Kde']:
            self.ob_dists = [gaussian_kde(wb[:, i]) for i in range(self.TS)]

            # increase the bandwidth a little
            for kde in self.ob_dists:
                kde.set_bandwidth(2 * kde.factor)
        else:
            raise ValueError('dist_type must either be "kde" or "gaussian"')

        self.dist_type = dist_type

        # SET PRIORS HERE FOR NOW (mean, standard deviation)

        self.priors = [Normal(5 * 60, 4 * 60),          # q       Mmol / min
                       Normal(3 * 60, 1.6 * 60),        # D 3e-12 m^2 / s   -> 200 µm^2 / min
                       Normal(30, 16),                   # tau     s
                       Normal(4, 2),                    # R_0     Mmol / m^2
                       Normal(4, 2),                    # kappa   Mmol / m^2
                       Normal(4, 8),                    # m      m^2 / mol
                       Normal(0.001, 0.0005)]          # b_0     None

    def set_bandwidth(self):
        """
        Run this function after instantiating the inference object to
        manually set the bandwidth for the observed bias KDE. Brings
        up an interactive matplotlib window.
        """

        for i in range(self.TS):

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title('Set the KDE bandwidth for the t={}, r={} cluster'.format(self.ts[i][0], self.ts[i][1]))

            # Adjust the subplots region to leave some space for the sliders and buttons
            fig.subplots_adjust(bottom=0.25)
            plt.hist(self.wb[:, i], bins=50, density=True, color='#94D0FF')

            x = np.linspace(min(self.wb[:, i]), max(self.wb[:, i]), 250)
            [line] = ax.plot(x, self.ob_dists[i](x))

            slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            slider = Slider(slider_ax, 'Bandwidth', 0.02, 0.5, valinit=self.ob_dists[i].factor)

            def kde_pdf(bw):
                self.ob_dists[i].set_bandwidth(bw)
                return self.ob_dists[i](x)

            def sliders_on_changed(val):
                line.set_ydata(kde_pdf(slider.val))
                fig.canvas.draw_idle()

            slider.on_changed(sliders_on_changed)
            reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
            reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')

            def reset_button_on_clicked(mouse_event):
                slider.reset()

            reset_button.on_clicked(reset_button_on_clicked)

            ok_button_ax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
            ok_button = Button(ok_button_ax, 'OK', hovercolor='0.975')

            def ok_button_on_clicked(mouse_event):
                plt.close()

            ok_button.on_clicked(ok_button_on_clicked)
            plt.show()

    def most_probable(self, method: str = 'SLSQP', seed: int = 0):
        '''
        Use scipy optimiser to find the most probable set of parameters

        Params:

            method:      which scipy optimize method to use

        Returns:

            best_params: a np.array of the most probable (w, p, b) parameters
        '''

        np.random.seed(seed)

        z = basinhopping(lambda x: -self.log_likelihood(x) - self.log_prior(x),
                     x0=np.array([225, 62, 17, 3, 5, 2.5, 0.0007]),
                     minimizer_kwargs={'method':method}, disp=True)

        return z.x

    def log_likelihood(self, params: Iterable):
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
            if self.dist_type == 'kde':
                return sum([self.ob_dists[i].logpdf(observed_bias(params, r, t))[0] for i, (t, r) in enumerate(self.ts)])
            else:
                return self.ob_dists.logpdf(observed_bias(params, self.r, self.t))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: Iterable):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])

    def infer(self, init_params: Iterable,
              n_steps: int=10000,
              burn_in: int=3000,
              seed: int=0) -> np.ndarray:
        """
        Given a set of starting parameters, perform MCMC bayesian inference for n_steps with
        a burn in of burn_in and a maximum step size in each direction of step_size

        Params:

            w0:        The starting point for w
            p0:        The starting point for p
            b0:        The starting point for b
            step:      Each new point is proposed as x' = x + uniform(-step, step) in each dimension
            n_steps:   The total number of MCMC steps to perform, after burn in
            burn_in:   The total number of burn in steps
            seed:      The number to to seed all random MC operations

        Returns:

            params:    np.array, shape: (n_steps, 3) -> the distribution over parameters

        """


        np.random.seed(seed)
        n = n_steps + burn_in
        L0 = self.log_likelihood(init_params) + self.log_prior(init_params)
        params = np.array(init_params)
        n_params = len(params)
        params_out = np.zeros((n, n_params))
        step = np.array([self.priors[i].sig / 20 for i in range(n_params)])
        total_accepts = 0
        rolling_accepts = 0
        rolling_rejects = 0
        t0 = time.time()

        for i in range(n):

            # add random purturbation
            params_ = params + np.random.normal(0, step)

            # evaluate the log-likelihood of our proposed move
            L_p = self.log_likelihood(params_) + self.log_prior(params_)

            # evaluate the probability of moving
            prob = np.exp(L_p - L0)

            # decide whether to move
            if np.random.uniform(0, 1) < prob:
                params = params_
                L0 = L_p
                total_accepts += 1
                rolling_accepts += 1
                rolling_rejects = 0
            else:
                rolling_rejects += 1

            if rolling_rejects == 100:
                print(' WARNING: {} simultaneous rejections'.format(100))
                rolling_rejects = 0

            # append params regardless of whether the step was accepted or not
            params_out[i, :] = params

            # increase or decrease step size to maintain optimum acceptance rate
            if i % 500 == 499:
                step = params_out[:i, :].std(0) / 4
                t = time.time() - t0
                print('\r' + 'Estimated time remaining : {:.0f}s. Total acceptance Rate: {:.3f}. Rolling acceptance rate: {:.3f}'.format(t * n / i - t, total_accepts / i, rolling_accepts / 500), end='')
                rolling_accepts = 0
        print('\r' + 'Acceptance Rate: {:.4f}'.format(total_accepts / n))

        return params_out[burn_in:, :]


if __name__ == '__main__':

    exp1 = True
    exp2 = False

    if exp1:

        import time

        np.set_printoptions(precision=8, suppress=True, linewidth=500)

        ts1 = [(10*60, 50e-6), (10*60, 200e-6), (10*60, 350e-6),
               (25*60, 50e-6), (25*60, 200e-6), (25*60, 350e-6),
               (40*60, 50e-6), (40*60, 200e-6), (40*60, 350e-6)]      # (seconds, m)

        ts2 = [(10*60, 50), (10*60, 200), (10*60, 350),
               (25*60, 50), (25*60, 200), (25*60, 350),
               (40*60, 50), (40*60, 200), (40*60, 350)]  # (seconds, µm)

        ts3 = [(10, 50), (10, 200), (10, 350),
               (25, 50), (25, 200), (25, 350),
               (40, 50), (40, 200), (40, 350)]  # (minutes, µm)

        ts = ts3
        wbs = [0.150, 0.040, 0.040,
               0.070, 0.015, 0.001,
               0.060, 0.010, 0.001]

        sca = [0.035, 0.020, 0.0150,
               0.025, 0.004, 0.0003,
               0.025, 0.003, 0.0003]

        sca = [0.035, 0.020, 0.0150,
               0.025, 0.004, 0.0003,
               0.025, 0.003, 0.0003]

        tr = np.array(ts)

        t, r = tr[:, 0], tr[:, 1]

        wb = np.random.normal(loc=wbs, scale=sca, size=(15000, 9))
        AD = AttractantDynamics(wb, ts, dist_type='gaussian')

        init_params = [AD.priors[i].mu for i in range(7)]

        t0 = time.time()
        ps = AD.infer(init_params, n_steps=500000, burn_in=500000)
        # mp = AD.most_probable()
        # print(mp)
        # print(np.concatenate([np.array(wbs)[:, None], concentration(mp, r, t)[:, None], concentration(init_params, r, t)[:, None]], axis=1))
        t1 = time.time()

        print('Inference completed in {:.2f}s'.format(t1-t0))

        # print(AD.log_likelihood(init_params), AD.log_prior(init_params))
        # print(AD.log_likelihood(mp), AD.log_prior(mp))

        # plot = False
        plot = True

        if plot:

            for i, name in enumerate(['q', 'D', 'tau', 'R_0', 'kappa', 'm', 'b_0']):

                plt.figure()
                plt.hist(ps[:, i], bins=100, density=True)
                AD.priors[i].plot()
                plt.title(name)

            plt.show()

    if exp2:

        ts = [(10, 50), (10, 200), (10, 350),
               (25, 50), (25, 200), (25, 350),
               (40, 50), (40, 200), (40, 350)]  # (minutes, µm)

        params = [200, 150, 15, 5.5, 6.5, 10.5, 0.001]
        wbs = np.array([observed_bias(params, r, t) for t, r in ts])
        print(wbs)
        # wbs = [0.2232, 0.0086, 0.001,  0.0256, 0.023,  0.0038, 0.0076, 0.0131, 0.006]
        sca = [0.05,   0.002,  0.0003, 0.008,  0.008,  0.001,  0.002,  0.004,  0.002]

        wb = np.random.normal(loc=wbs, scale=sca, size=(10000, 9))
        AD = AttractantDynamics(wb, ts)
        # AD.set_bandwidth()

        init_params = [AD.priors[i].mu for i in range(7)]

        t0 = time.time()
        ps = AD.infer(init_params, n_steps=500000, burn_in=500000)
        t1 = time.time()

        print('Inference completed in {:.2f}s'.format(t1-t0))

        for i, name in enumerate(['q', 'D', 'tau', 'R_0', 'kappa', 'm', 'b_0']):
            plt.figure()
            plt.hist(ps[:, i], bins=100, density=True)
            AD.priors[i].plot()
            plt.title(name)

        plt.show()