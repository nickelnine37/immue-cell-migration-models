import numpy as np
from in_silico.sources import Source
from utils.distributions import Bernoulli, WrappedNormal, TruncatedNormal
from utils.angles import angle_between

# use negative y-axis as the reference axis
reference_axis = np.array([0, -1])

class Leukocyte:

    def __init__(self, *args, **kwargs):
        pass

    def walk(self, X0: np.array, T: int):
        raise NotImplementedError

    def multi_walk(self, X0s: list, T:int):
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
                 w: float,
                 p: float,
                 b: float,
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

        self.w = w
        self.p = p
        self.b = b
        self.source = source
        self.s = s
        self.step = TruncatedNormal(mu=0, sig=self.s)

    def walk(self, X0: np.array, T: int) -> np.array:
        """
        Perform a biased-persistent random walk from starting point X0, for T time steps

        Params:

            X0:       The starting coordinates for the leukocyte
            T:        The number of steps to move through

        Returns:

            np.array (T+1, 2): the x-y coordinates of the path traversed beginning at X0

        """

        path = np.zeros((T + 1, 2))
        path[0, :] = X0
        biased = Bernoulli(mu=self.w).sample(size=T)  # pre-compute all the b/p decisions in vector form

        for i, bias in enumerate(biased):

            if bias or i == 0:
                sig = -2 * np.log(self.b)
                dir_to_source = self.source.direction_to_source(path[i])
                mu = angle_between(reference_axis, dir_to_source)

            else:
                sig = -2 * np.log(self.p)
                mu = previous_angle

            if np.isinf(sig):
                sig = 100

            alpha = WrappedNormal(mu=mu, sig=sig).sample()
            previous_angle = alpha
            s_t = self.step.sample()
            dx = s_t * np.sin(alpha)
            dy = s_t * np.cos(alpha)
            path[i + 1, 0] = path[i, 0] + dx
            path[i + 1, 1] = path[i, 1] - dy

        return path



