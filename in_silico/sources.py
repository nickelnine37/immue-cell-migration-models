import numpy as np
from scipy.special import expi
from typing import Union, Iterable

class Source:

    def __init__(self, position: np.array):
        self.position = position

    def direction_to_source(self, coords: tuple):
        raise NotImplementedError


class PointSource(Source):

    def __init__(self, position: np.array=np.array([0, 0])):
        super().__init__(position)

    def direction_to_source(self, coords: np.array):
        return self.position - coords

class CircularSource(Source):

    def __init__(self, position: np.array=np.array([0, 0]), radius: float=1):
        super().__init__(position)
        self.radius = radius

    def direction_to_source(self, coords: np.array):
        dists = ((self.position - coords) ** 2).sum(0) ** 0.5
        out = self.position - coords
        out[dists < self.radius] *= -1
        return out

class Wound:

    def __init__(self, position: np.array):
        self.position = position

    def concentration(self, params, x, y, t):
        raise NotImplementedError

class PointWound(Wound):

    def __init__(self, position: np.array=np.array([0, 0])):
        super().__init__(position)

    def concentration(self, params, x, y, t):
        r = ((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2) ** 0.5
        return A(params, r, t)

class CellsOnMarginWound(Wound):

    def __init__(self, position: np.array=np.array([0, 0]), n_cells=10, radius=1):
        super().__init__(position)
        self.radius = radius
        self.n_cells = n_cells
        self.positions = [(radius * np.cos(theta), radius * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, n_cells + 1)[:-1]]

    def concentration(self, params, x, y, t):
        rs = [((x - x_) ** 2 + (x - y_) ** 2) ** 0.5 for x_, y_ in self.positions]
        return sum([A(params, r, t) for r in rs])

def A(params: Iterable, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
    """
    This function returns the concentration of attractant at a
    radial distance r and time t for a continuous point source
    emitting attractand at a rate q from the origin from t=0
    to t=tau with diffusion constant D.

    Parameters
    ----------
    r       the radial distance from the origin. Can be float or array
    t       time
    tau     time attractant is released for
    q       flow rate of attractant
    D       diffusion constant

    Returns
    -------
    A       The attractant concentration

    """

    q, D, tau = params[:3]

    if isinstance(r, (float, int, np.ndarray)) and isinstance(t, (float, int)):

        if t < tau:
            return - (q / (4 * np.pi * D)) * expi(- r ** 2 / (4 * D * t))
        else:
            return (q / (4 * np.pi * D)) * (expi(- r ** 2 / (4 * D * (t - tau))) - expi(- r ** 2 / (4 * D * t)))

    elif isinstance(r, (float, int, np.ndarray)) and isinstance(t, np.ndarray):

        assert r.shape == t.shape, 'r and t must be the same shape, but they have shapes {} and {} respectively'.format(
            r.shape, t.shape)

        out = - expi(- r ** 2 / (4 * D * t))
        out[t > tau] += expi(- r[t > tau] ** 2 / (4 * D * (t[t > tau] - tau)))

        return (q / (4 * np.pi * D)) * out

    else:
        raise TypeError('r and t must both be floats/ints or numpy arrays')


