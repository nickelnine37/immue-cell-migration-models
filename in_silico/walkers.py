import numpy as np
from in_silico.sources import Source, StaticGaussian
from utils.distributions import bernoulli, wrapped_normal_sample, truncated_normal_sample
from utils.angles import angle_between

# THIS IS MY SUPER BASIC SOURCE DEFINITION
source_coords = np.array([0, 0])
reference_axis = np.array([0, -1]) # use negative y-axis as the reference axis, as per the image


def direction_to_source(coords):
    return source_coords - coords


def BP_walk(X0: np.array, T: int, w: float, p: float, b:float) -> np.array:
    """
    Perform a biased-persistent random walk from starting point X0, for T time steps
    """

    path = np.zeros((T + 1, 2))
    path[0, :] = X0
    biased = bernoulli(w, T)

    sig_b = -2 * np.log(b)
    sig_p = -2 * np.log(p)

    for i, bias in enumerate(biased):
        current_position = path[i]
        dir_to_source = direction_to_source(current_position)

        if bias or i == 0:
            sig = sig_b
            mu = angle_between(reference_axis, dir_to_source)
        else:
            sig = sig_p
            mu = previous_angle
        if np.isinf(sig):
            sig = 100

        alpha = wrapped_normal_sample(mu, sig)
        previous_angle = alpha
        s = truncated_normal_sample(0.2)
        dx = s * np.sin(alpha)
        dy = s * np.cos(alpha)
        path[i + 1, 0] = path[i, 0] + dx
        path[i + 1, 1] = path[i, 1] - dy  # -ve because (0, -1) reference vector??

    return path
