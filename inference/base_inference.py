from typing import Iterable, Union
import time
import numpy as np
from tqdm import tqdm
from utils.parallel import parallel_methods

class inferer:

    def __init__(self, *args, **kwargs):
        self.priors = None

    def log_likelihood(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def log_prior(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def infer(self, n_steps: int, burn_in: int, seed: int=0,
              suppress_warnings: bool=False, use_tqdm: bool=True) -> np.ndarray:
        """
        Perform one session of MCMC inference on biased-persistent parameters.
        The starting point is sampled from the priors.

        Parameters
        ----------
        n_steps             The number of steps, after burn_in to perform
        burn_in             The number of burn in steps
        seed                A random seed - useful for multiprocessing
        suppress_warnings   Whether to suppress warnings about failure rates
        use_tqdm            Whether to use a tqdm progress bar

        Returns
        -------

        params_out          A (n_steps, n_dimensions) array giving all the sampled posterior values

        """

        np.random.seed(seed)

        # sample the initial parameters from the priors
        init_params = np.array([prior.sample() for prior in self.priors])

        # get the initial log likelihood and log prior values
        L0 = self.log_likelihood(init_params) + self.log_prior(init_params)

        # calculate some values for the inference
        n = n_steps + burn_in
        check_every = n_steps // 100
        params = np.array(init_params)
        n_params = len(params)
        params_out = np.zeros((n, n_params))

        # initialise the step size to be 1/20 of the std of the priors
        step = [prior.std() / 20 for prior in self.priors]
        step_factor = 1 # for adapting step size towards optimal acceptance ratio
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
                    print(' WARNING: 100 simultaneous rejections')
                rolling_rejects = 0

            # append the params regardless of whether the step was accepted or not
            params_out[i, :] = params

            # adapt the gaussian kernel width to be 1/3 the standard deviation of previous steps
            if i % check_every == check_every - 1:
                ar, rar = total_accepts / i, rolling_accepts / check_every

                # drive acceptance rate towards 0.234, as per Roberts, G.O., Gelman, A., Gilks, W.R. (1997). Weak Convergence and Optimal Scalingof Random Walk Metropolis Algorithms.Ann. Appl. Probab.7, 110-20. Though note there's been some debate since e.g.  http://probability.ca/jeff/ftpdir/mylene2.pdf
                step_factor += (ar - 0.234)
                step = step_factor * np.array(params_out)[:i, ].std(0) / 3

                if use_tqdm:
                    pbar.set_description('Total acceptance Rate: {:.3f}. Rolling acceptance rate: {:.3f}'.format(ar, rar))
                rolling_accepts = 0

        return np.array(params_out)[burn_in:, :]

    def multi_infer(self, n_walkers: int, n_steps: int, burn_in: int, seed: int=0,
                    suppress_warnings: bool=False, use_tqdm: bool=True) -> np.ndarray:
        """
        Perform inference with n_walkers starting points, in parallel

        Parameters
        ----------
        n_walkers           The number of parallel processes to run
        n_steps             The number of steps, after burn_in to perform
        burn_in             The number of burn in steps
        seed                A random seed - useful for multiprocessing
        suppress_warnings   Whether to suppress warnings about failure rates
        use_tqdm            Whether to use a tqdm progress bar

        Returns
        -------
        params_out          A (n_steps, n_dimensions) array giving all the sampled posterior values

        """

        np.random.seed(seed)
        objs = [self for _ in range(n_walkers)]
        methods = ['infer' for _ in range(n_walkers)]

        params = [{'n_steps': n_steps,
                   'burn_in': burn_in,
                   'seed': i + seed,
                   'use_tqdm': use_tqdm,
                   'suppress_warnings': suppress_warnings} for i in range(n_walkers)]

        t0 = time.time()
        print('Beginning MCMC walk in parallel')
        res = parallel_methods(objs, methods, params, backend='multiprocessing')
        print('Completed MCMC walk in {:.2f}s'.format(time.time() - t0))
        try:
            return np.concatenate(res, axis=0)
        except ValueError:
            print('Warning: one or more of the parallel MCMC walkers failed')
            return np.concatenate([r for r in res if r.shape == (n_steps, len(self.priors))], axis=0)
