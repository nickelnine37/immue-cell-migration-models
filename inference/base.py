from typing import Iterable, Union
import time
import numpy as np
from tqdm import tqdm
from utils.parallel import parallel_methods

class MCMC:

    def __init__(self, *args, **kwargs):
        self.priors = None

    def log_likelihood(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def log_prior(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def infer(self, init_params: np.ndarray, n_steps: int, burn_in: int,
              seed: int=0, suppress_warnings: bool=False, use_tqdm: bool=True) -> np.ndarray:

        np.random.seed(seed)

        n = n_steps + burn_in
        check_every = n_steps // 100

        L0 = self.log_likelihood(init_params) + self.log_prior(init_params)
        params = np.array(init_params)
        n_params = len(params)
        params_out = np.zeros((n, n_params))

        step = [prior.std() / 20 for prior in self.priors]
        total_accepts = 0
        rolling_accepts = 0
        rolling_rejects = 0

        if use_tqdm:
            pbar = tqdm(range(n))
        else:
            pbar = range(n)

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

            if rolling_rejects == 100:
                if not suppress_warnings:
                    print(' WARNING: 100 simultaneous rejections')
                rolling_rejects = 0

            # append the params regardless of whether the step was accepted or not
            params_out[i, :] = params

            # adapt the guassian kernel width to be half the standard deviation of previous steps
            if i % check_every == check_every - 1:
                step = np.array(params_out)[:i, ].std(0) / 3
                if use_tqdm:
                    ar, rar = total_accepts / i, rolling_accepts / check_every
                    pbar.set_description('Total acceptance Rate: {:.3f}. Rolling acceptance rate: {:.3f}'.format(ar, rar))
                rolling_accepts = 0

        return np.array(params_out)[burn_in:, :]

    def multi_infer(self, init_params: np.ndarray, n_steps: int, burn_in: int,
                    seed: int=0, suppress_warnings: bool=False, use_tqdm: bool=True) -> np.ndarray:

        np.random.seed(seed)
        n_walkers = init_params.shape[0]
        n_params = init_params.shape[1]
        objs = [self for _ in range(n_walkers)]
        methods = ['infer' for _ in range(n_walkers)]

        params = [{'init_params': init_params[i, :], 'n_steps': n_steps, 'burn_in': burn_in, 'seed': i + seed, 'use_tqdm': use_tqdm, 'suppress_warnings': suppress_warnings} for i in range(n_walkers)]

        t0 = time.time()
        print('Beginning MCMC walk in parallel')
        res = parallel_methods(objs, methods, params, backend='multiprocessing')
        print('Completed MCMC walk in {:.2f}s'.format(time.time() - t0))
        try:
            return np.concatenate(res, axis=0)
        except ValueError:
            print('Warning: one or more of the parallel MCMC walkers failed')
            return np.concatenate([r for r in res if r.shape == (n_steps, n_params)], axis=0)
