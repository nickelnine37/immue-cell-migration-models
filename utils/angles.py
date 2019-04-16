import numpy as np


def angle_between(v1, v2):
    """
    v1 is the reference vector. Angle is positive anti-clockwise, negative clockwise
    between -pi and pi.
    """
    dot = v1 @ v2
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(det, dot)

def convert_path(paths: np.array, reference_axis: np.array, direction_to_source: object):
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

    d_to_sources = np.apply_along_axis(direction_to_source, 1, paths)
    betas = np.apply_along_axis(lambda d: angle_between(reference_axis, d), 1, d_to_sources)[:-1, :]

    return alphas, betas