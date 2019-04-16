import numpy as np
from typing import Union

# ------- GLOBALS ----------

# for wrapped distributions, keep wrapping intil the change is less than tol
tol = 1e-10

def wrapped_normal_sample(mu: float, sig: float, size: Union[tuple, float]=None) -> Union[np.array, float]:
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

def truncated_normal_sample(sig: float, size=None) -> Union[np.array, float]:
    """
    Return an array of independant draws from a truncated normal distribution. Note that
    the passed standard deviation sig is the would-be standard deviation from a non-truncated
    distribution. The true standard deviation is different.
    """
    return np.abs(np.random.normal(0, sig, size))

def normal_pdf(x: np.array, mu: np.array, sig: np.array) -> np.array:
    """
    Get the value of the normal distribution pdf function at point x.
    """
    return (1 / (sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - mu) ** 2 / (2 * sig ** 2))


def wrapped_normal_pdf(x: float, mu: float, sig: float) -> float:
    """
    First attempt at a pdf function for a wrapped normal distribution.
    Can only take scalar inputs, e.i. evaluate points f(x) one by one.

    Takes 10-60 microseconds to run
    """

    assert isinstance(x, float) and isinstance(mu, float), 'Cannot pass arrays- must be floats'
    assert isinstance(sig, float), 'Sigma must currently be a float'

    wrap_width = 2 * np.pi

    # if sigma is greater than 6, the difference between it and a uniform distribution is ~1e-8
    if sig > 6:
        return 1 / wrap_width

    i = 1
    value = normal_pdf(x, mu, sig)  # value of the function at x in the first loop
    while True:
        fs = normal_pdf(np.array([x + i * wrap_width, x - i * wrap_width]), mu, sig).sum()
        value += fs

        # this tolerence is defined at the start
        if fs < tol / 2:
            break
        i += 1

    return value


def wrapped_normal_pdf2(x: np.array, mu: np.array, sig: float) -> np.array:
    """
    A vectorised version -- still testing. Also works for floats
    """


    # as before. For this reason, sig has to be a float still
    if sig > 6:
        return 1 / (2 * np.pi)

    # blindly opt for 5 loops either side
    return normal_pdf(np.array([x + 2 * np.pi * i for i in range(-4, 5)]), mu, sig).sum(0)

if __name__ == '__main__':

    values = np.array([1, 2, 3], dtype='float')
    means = np.array([0, 1, -1], dtype='float')
    sig = 1.3

    c = np.array([wrapped_normal_pdf(value, mean, sig) for value, mean in zip(values, means)])
    d = np.array([wrapped_normal_pdf2(value, mean, sig) for value, mean in zip(values, means)])
    e = wrapped_normal_pdf2(values, means, sig)



