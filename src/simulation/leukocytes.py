import numpy as np
import pandas as pd
from simulation.sources import Source
from utils.distributions import Bernoulli, WrappedNormal, TruncatedNormal
from utils.numpy import angle_between
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

        if (np.array(params) > 1).any() or (np.array(params) < 0).any():
            raise ArgumentError('The values for p, b and w must be between 0 and 1, but they are p={}, b={}, w={}'.format(p, b, w))

        self.p = p if p != 0 else 0.01
        self.b = b if b != 0 else 0.01
        self.sigma_p = (-2 * np.log(p)) ** 0.5
        self.sigma_b = (-2 * np.log(b)) ** 0.5

        self.source = source
        self.s = s
        self.step = TruncatedNormal(sig=self.s)

    def walk(self, X0s: np.ndarray, T: int):

        N = X0s.shape[0]
        path = np.zeros((T + 1, 2, N))
        path[0, :, :] = X0s.T
        bias_decisions = Bernoulli(mu=self.w).sample(size=(T, N))  # pre-compute all the b/p decisions in vector form
        bias_decisions[0, :] = True
        steps = self.step.sample(size=T)

        for t in range(T):

            alpha = np.zeros(N)

            biased = bias_decisions[t, :]
            dir_to_source = self.source.direction_to_source(path[t, :, :].T)
            beta = angle_between(reference_axis, dir_to_source.T)
            bias_angles = WrappedNormal(mu=beta, sig=self.sigma_b).sample(N)
            alpha[biased] = bias_angles[biased]

            if t > 0:
                persis_anlges = WrappedNormal(mu=previous_angles, sig=self.sigma_p).sample(N)
                alpha[~biased] = persis_anlges[~biased]

            previous_angles = alpha
            path[t + 1, :, :] = path[t, :, :] + steps[t] * np.array([np.sin(alpha), -np.cos(alpha)])

        # return path

        a = [np.concatenate([np.array([((i + 1) * np.ones(T+1)).astype(int), np.arange(T+1).astype(int)]).T, path[:, :, i]], axis=1) for i in range(N)]
        return pd.DataFrame(np.concatenate(a, axis=0), columns=['trackID', 'time', 'x', 'y'])



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from simulation.sources import PointSource
    from utils.plotting import plot_paths

    # set some parameters
    w, p, b = 0.3, 0.5, 0.7
    preset_params = np.array([w, p, b])

    # generate some paths with these parameters, with two different sources
    source1 = PointSource(position=(0, 0))
    source2 = PointSource(position=(1.5, -1.5))
    leukocytes1 = BP_Leukocyte(params=preset_params, source=source1)
    leukocytes2 = BP_Leukocyte(params=preset_params, source=source2)

    n_walkers = 20
    n_frames = 100
    paths1 = leukocytes1.walk(X0s=np.random.uniform(-5, 5, size=(n_walkers, 2)), T=n_frames)
    paths2 = leukocytes2.walk(X0s=np.random.uniform(-5, 5, size=(n_walkers, 2)), T=n_frames)

    # plot the paths
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plot_paths(paths1, source1, ax1)
    plot_paths(paths2, source2, ax2)
    ax1.set_title('Source 1')
    ax2.set_title('Source 2')
    plt.show()
