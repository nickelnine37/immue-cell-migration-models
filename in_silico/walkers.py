import numpy as np
from in_silico.sources import Source
from utils.distributions import Bernoulli, WrappedNormal, TruncatedNormal
from utils.angles import angle_between
from utils.exceptions import ArgumentError

# use negative y-axis as the reference axis
reference_axis = np.array([0, -1])

class Leukocyte:

    def __init__(self, *args, **kwargs):
        pass

    def walk(self, X0: np.array, T: int):
        raise NotImplementedError

    def multi_walk(self, X0s: np.ndarray, T:int):
        """
        For a list of initial coordinates, X0, perform len(X0) walks for T steps
        and concatenate the results

        Params:
            X0:     list of np.arrays, defining the starting coordinates
            T:      the number of steps to perform

        Returns:
            paths:  a (T, 2, N) np.array with all the coordinates of the walk
        """
        return np.concatenate([self.walk(X0, T)[:, :, None] for X0 in X0s], axis=2)


class BP_Leukocyte(Leukocyte):

    def __init__(self,
                 params: np.ndarray,
                 source: Source,
                 s: float=0.2) -> None:
        """
        Initialise a class of biased-persistent Leukocytes in an environment with
        a source.

        Params:
                w:        The variable w which defines the probability of taking a biased step
                p:        The variable p, defining the variance of persistent motion
                b:        The variable b, defining the variance of boased motion
                source:   A source.Source, with a direction_to_source method
                s:        The sigma parameter for the truncated normal step

        """

        super().__init__()

        w, p, b = params

        self.w = w

        if any([p < 0, p > 1, b < 0, b > 1, w < 0, w > 1]):
            raise ArgumentError('The values for p, b and w must be between 0 and 1, but they are p={}, b={}, w={}'.format(p, b, w))

        self.p = p if p != 0 else 0.01
        self.b = b if b != 0 else 0.01
        self.source = source
        self.s = s
        self.step = TruncatedNormal(sig=self.s)

    def walk(self, X0s: np.ndarray, T: int):

        N = X0s.shape[0]
        path = np.zeros((T + 1, 2, N))

        path[0, :, :] = X0s.T

        bias_decisions = Bernoulli(mu=self.w).sample(size=(T, N))  # pre-compute all the b/p decisions in vector form
        bias_decisions[0, :] = True
        sigmas = np.zeros((T, N))
        sigmas[bias_decisions] = self.b
        sigmas[~bias_decisions] = self.p
        sigmas = - 2 * np.log(sigmas)
        sigmas[np.isinf(sigmas)] = 100
        steps = self.step.sample(size=T)

        for t in range(T):

            biased = bias_decisions[t, :]
            dir_to_source = self.source.direction_to_source(path[t, :, :].T)
            mu = angle_between(reference_axis, dir_to_source.T)

            if t > 0:
                mu[~biased] = previous_angles[~biased]

            alpha = WrappedNormal(mu=mu, sig=sigmas[t, :]).sample(N)
            previous_angles = alpha
            path[t + 1, :, :] = path[t, :, :] + steps[t] * np.array([np.sin(alpha), -np.cos(alpha)])

        return path


if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    from in_silico.sources import PointSource

    N = 20
    B = BP_Leukocyte(np.array([0.5, 0.8, 0.8]), PointSource(np.array([3, 3])))

    t0 = time.time()
    paths = B.walk(X0s=np.random.uniform(-5, 5, size=(N, 2)), T=100)
    t2 = time.time()

    from plotting import plot_paths

    plot_paths(paths, PointSource(np.array([3, 3])))
