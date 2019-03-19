import numpy as np
from typing import Union

def wrapped_normal(mu: float, sig: float, size: Union[tuple, float]=None) -> Union[np.array, float]:
    """
    Return an array of independant draws from a wrapped normal distribution
    """
    return np.mod(np.random.normal(np.pi + mu, sig, size), 2 * np.pi) - np.pi

def bernoulli(w: float, size: Union[tuple, float]=None) -> Union[np.array, float]:
    """
    Return a boolean array of independant draws from a Bernoulli distribution
    """
    out = np.random.uniform(0, 1, size)
    return out < w

def truncated_normal(sig: float, size=None) -> Union[np.array, float]:
    """
    Return an array of independant draws from a truncated normal distribution. Note that
    the passed standard deviation sig is the would-be standard deviation from a non-truncated
    distribution. The true standard deviation is different.
    """
    return np.abs(np.random.normal(0, sig, size))