import numpy as np
from utils.angles import angle_between
from walkers import reference_axis

from utils.distributions import wrapped_normal_pdf, wrapped_normal_pdf2



class BPbayesian:

    def __init__(self, paths: np.array, direction_to_source: object):

        """
        Params:

            paths:                np.array - (T+1, 2, N) - paths of N walkers walking for T timesteps
            direction_to_source:  a function that takes array([x, y]) and returns the direction to the source.

        """

        self.paths = paths
        self.direction_to_source = direction_to_source
        alphas, betas = self.convert_path(paths)
        self.alphas = alphas
        self.betas = betas

    def convert_path(self, paths: np.array):
        """
        Given a set of raw x-y coordinates, get a list of the alphas and betas

        Params:

            paths:                np.array - (T+1, 2, N) - paths of N walkers walking for T timesteps
            reference_axis:       np.array - (2, ) direction which angles should be measured around
            direction_to_source:  a function that takes array([x, y]) and returns the direction to the source.

        Returns:

            alphas:    np.array - (T, N): angle taken at each time step
            betas:     np.array - (T, N): direction to source at each time step

        """

        moves = paths[1:, :, :] - paths[:-1, :, :]
        alphas = np.apply_along_axis(lambda move: angle_between(reference_axis, move), 1, moves)
        d_to_sources = np.apply_along_axis(self.direction_to_source, 1, paths)
        betas = np.apply_along_axis(lambda d: angle_between(reference_axis, d), 1, d_to_sources)[:-1, :]

        return alphas, betas

    def L1(self, w, p, b):
        """
        Get the likelihood of
        """

        sig_b = -2 * np.log(b)
        sig_p = -2 * np.log(p)
        ps = 0

        for walker in range(self.alphas.shape[1]):

            alphas = self.alphas[:, walker]
            betas = self.betas[:, walker]

            ps += np.log(wrapped_normal_pdf(alphas[0], betas[0], sig_b))

            for alpha, beta, alpha_prev in zip(alphas[1:], betas[1:], alphas[:-1]):
                p_t = w * wrapped_normal_pdf(alpha, beta, sig_b) + (1 - w) * wrapped_normal_pdf(alpha, alpha_prev, sig_p)
                ps += np.log(p_t)

        return ps

    def L2(self, w, p, b):

        sig_b = -2 * np.log(b)
        sig_p = -2 * np.log(p)
        ps = 0

        for walker in range(self.alphas.shape[1]):

            alphas = self.alphas[1:, walker]
            alpha_prevs = self.alphas[:-1, walker]
            betas = self.betas[1:, walker]

            p_0 = wrapped_normal_pdf2(self.alphas[0, walker], self.betas[0, walker], sig_b)
            p_t = w * wrapped_normal_pdf2(alphas, betas, sig_b) + (1 - w) * wrapped_normal_pdf2(alphas, alpha_prevs, sig_p)

            ps += np.log(p_0)
            ps += np.log(p_t).sum()

        return ps

    def infer(self, w0, p0, b0,
              step=0.02,
              n_steps=10000,
              burn_in=3000):

        params = []
        w, p, b = w0, p0, b0
        L0 = self.L2(w, p, b)

        for i in range(n_steps):

            dw = np.random.uniform(-step, step)
            dp = np.random.uniform(-step, step)
            db = np.random.uniform(-step, step)

            w_, p_, b_ = w + dw, p + dp, b + db

            if any([a_ > 1 or a_ < 0 for a_ in [w_, b_, p_]]):
                continue

            L_p = self.L2(w_, p_, b_)
            prob = np.exp(L_p - L0)

            if np.random.uniform(0, 1) < prob:
                w, p, b = w_, p_, b_
                L0 = L_p
                if i > burn_in:
                    params.append([w, p, b])

        return np.array(params)

if __name__ == '__main__':

    from walkers import BP_walk
    from walkers import reference_axis, direction_to_source
    import time
    import matplotlib.pyplot as plt


    paths = np.concatenate([BP_walk(np.array([i, i]), 100, 0.3, 0.7, 0.7)[:, :, None] for i in range(10)], axis=2)

    BI = BPbayesian(paths, direction_to_source)

    t0 = time.time()
    print('starting inference 1')
    t1 = time.time()
    print('starting inference 2')
    params2 = BI.infer(w0=0.3, p0=0.6, b0=0.4,
                        n_steps=10000,
                        burn_in=3000)
    t2 = time.time()

    print('L1 : {:.2f}s'.format(t1 - t0))
    print('L2 : {:.2f}s'.format(t2 - t1))

    def plot(params, title=None):

        plt.figure()
        stds = np.std(params, 0)
        means = np.mean(params, 0)

        for i, typ in enumerate(['w', 'p', 'b']):
            plt.hist(params[:, i], label='${}$ = {:.2f} $\pm$ {:.2f}'.format(typ, means[i], stds[i]), bins=100, alpha=0.7)

        plt.legend()
        plt.xlim(0, 1)
        plt.title(title)
        plt.show()

    # plot(params1, title='Looped')
    plot(params2, title='Vectorised')

