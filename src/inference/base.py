from typing import Iterable, Union
import time
import numpy as np
from tqdm import tqdm
from utils.parallel import parallel_methods

class Inferer:
    """
    This is the base inference class that both the walker and attractant inference inherit from.
    It implements the methods necessary for sampling from the posterior using a simple MCMC
    procedure.
    """

    def __init__(self, *args, **kwargs):
        self.priors = None

    def log_likelihood(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def log_prior(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def MC_chain(self,
                 n_steps: int,
                 burn_in: int,
                 target_ar: float=0.234,
                 seed: int=0,
                 suppress_warnings: bool=False,
                 use_tqdm: bool=True) -> np.ndarray:
        """
        Perform one session of MCMC inference on biased-persistent parameters.
        The starting point is sampled from the priors.

        Parameters
        ----------
        n_steps             The number of steps, after burn_in to perform
        burn_in             The number of burn in steps
        target_ar           The target acceptance rate - must be between 0 ad 1
        seed                A random seed - useful for multiprocessing
        suppress_warnings   Whether to suppress warnings about failure rates
        use_tqdm            Whether to use a tqdm progress bar

        Returns
        -------

        params_out          A (n_steps, n_dimensions) array giving all the sampled posterior values

        """

        np.random.seed(seed)
        assert 0 < target_ar < 1, 'Target acceptance rate must be between 0 and 1'

        # sample the initial parameters from the priors
        init_params = np.array([prior.sample() for prior in self.priors])

        # get the initial log likelihood and log prior values
        L0 = self.log_likelihood(init_params) + self.log_prior(init_params)

        # calculate some values for the inference
        step_factor = 1
        n = n_steps + burn_in
        check_every = n_steps // 100
        params = np.array(init_params)
        params_out = np.zeros((n, len(params)))

        # initialise the step size to be 1/20 of the std of the priors
        step = [prior.std() / 20 for prior in self.priors]
        total_accepts = 0
        rolling_accepts = 0
        rolling_rejects = 0

        # set up a tqdm progress bar
        if use_tqdm:
            pbar = tqdm(range(n))
        else:
            pbar = range(n)

        # iterate through the inference steps
        for i in pbar:

            # add random purturbation to w, p, b
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

            # count rolling rejects and warn if necessary
            if rolling_rejects == 100:
                if not suppress_warnings:
                    print('WARNING: 100 simultaneous rejections')
                rolling_rejects = 0

            # append the params regardless of whether the step was accepted or not
            params_out[i, :] = params

            # adapt the guassian kernel width to be 1/3 the standard deviation of previous steps
            if i % check_every == check_every - 1:
                ar, rar = total_accepts / i, rolling_accepts / check_every

                # drive acceptance rate towards target acceptance rate
                step_factor += (ar - target_ar)
                if step_factor < 0:
                    step_factor = 0.01
                step = step_factor * np.array(params_out)[:i, ].std(0) / 3

                if use_tqdm:
                    pbar.set_description('Total acceptance Rate: {:.3f}. Rolling acceptance rate: {:.3f}'.format(ar, rar))
                rolling_accepts = 0

        return np.array(params_out)[burn_in:, :]

    def infer(self,
              n_chains: int,
              n_steps: int,
              burn_in: int,
              target_ar: float=0.234,
              seed: int=0,
              suppress_warnings: bool=False,
              progress_bar: bool=True) -> np.ndarray:
        """
        Perform inference with n_chains starting points, in parallel

        Parameters
        ----------
        n_chains            The number of parallel MCMC processes to run
        n_steps             The number of steps, after burn_in to perform
        burn_in             The number of burn in steps
        target_ar           The target acceptance rate - must be between 0 and 1
        seed                A random seed - useful for multiprocessing
        suppress_warnings   Whether to suppress warnings about failure rates
        use_tqdm            Whether to use a tqdm progress bar

        Returns
        -------
        params_out          A (n_steps * n_chains, n_dimensions) array giving all the sampled posterior values

        """

        np.random.seed(seed)
        objs = [self for _ in range(n_chains)]
        methods = ['MC_chain' for _ in range(n_chains)]

        params = [{'n_steps': n_steps,
                   'burn_in': burn_in,
                   'seed': i + seed,
                   'use_tqdm': progress_bar,
                   'suppress_warnings': suppress_warnings,
                   'target_ar': target_ar} for i in range(n_chains)]

        t0 = time.time()
        print('Beginning {} MCMC chains in parallel. Burn in: {}. Target acceptance rate: {}'.format(n_chains, burn_in, target_ar))
        res = parallel_methods(objs, methods, params, backend='multiprocessing')
        print('Sampling completed in {:.2f}s'.format(time.time() - t0))
        try:
            return np.concatenate(res, axis=0)
        except ValueError:
            print('Warning: one or more of the parallel MCMC walkers failed')
            return np.concatenate([r for r in res if r.shape == (n_steps, len(self.priors))], axis=0)

