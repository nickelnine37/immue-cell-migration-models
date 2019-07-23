import numpy as np
from scipy.special import expi
from typing import Union, Iterable

class Source:

    def __init__(self, position: np.array, units='pixels'):

        assert units in ['microns', 'pixels']
        self.position = position
        self.units = units
        self.x, self.y = self.position

    def direction_to_source(self, coords: tuple):
        raise NotImplementedError

    def to_microns(self, microns_per_pixel: float):
        if self.units is 'pixels':
            self.position *= microns_per_pixel
            self.units = 'microns'
            self.x, self.y = self.position
        else:
            print('Units are already microns')
        return self

    def to_pixels(self, microns_per_pixel: float):
        if self.units is 'microns':
            self.position /= microns_per_pixel
            self.units = 'pixels'
            self.x, self.y = self.position
        else:
            print('Units are already pixels')
        return self

class PointSource(Source):

    def __init__(self, position: np.array=np.array([0, 0]), units='pixels'):
        super().__init__(position, units)

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

class Wound(Source):

    def __init__(self, position: np.array, units='microns'):
        super().__init__(position)

    def concentration(self, params, r, t):
        raise NotImplementedError

class PointWound(Wound):

    def __init__(self, position: np.array=np.array([0, 0])):
        super().__init__(position)

    def concentration(self, params, r, t):
        return concentration(params, r, t)

class CellsOnWoundMargin(Wound):

    def __init__(self, position: np.array=np.array([0, 0]), n_cells=10, radius=15):
        super().__init__(position)
        self.radius = radius
        self.n_cells = n_cells
        self.positions = np.array([(radius * np.cos(theta), radius * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, n_cells + 1)[:-1]])
        self.x = self.positions[:, 0].reshape(-1, 1)
        self.y = self.positions[:, 1].reshape(-1, 1)

    def concentration(self, params: np.ndarray, r: np.ndarray, t: np.ndarray):

        rs = (self.x ** 2 + (r.reshape(-1)[None, :] - self.y) ** 2) ** 0.5
        return concentration(params, rs, np.repeat(t[None, :], len(self.x), axis=0)).sum(0).reshape(r.shape)

class CellsInsideWound(Wound):

    def __init__(self, position: np.array=np.array([0, 0]), n_cells=9, radius=15):
        super().__init__(position)
        self.radius = radius
        self.n_cells = n_cells
        self.positions = sunflower(n_cells)
        self.x = self.positions[:, 0].reshape(-1, 1)
        self.y = self.positions[:, 1].reshape(-1, 1)

    def concentration(self, params, r, t):

        rs = [(x ** 2 + (r - y) ** 2) ** 0.5 for x, y in self.positions]
        return sum([concentration(params, r, t) for r in rs])

def concentration(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
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


def sunflower(n):
    """
    Spread n points out in a fibonnaci spiral, inside a
    circle of radius 1.

    Parameters
    ----------
    n       the number of points

    Returns
    -------
    coords   a list of tuples holding the x-y coordinates of the n points
    """

    b = int(n ** 0.5)
    phi = (5 ** 0.5 + 1) / 2
    k = np.arange(1, n + 0.1, 1)
    r = (k - 0.5) ** 0.5 / (n - (b + 1) / 2) ** 0.5
    r[k > (n - b)] = 1
    theta = 2 * np.pi * k / phi ** 2
    return np.array(list(zip(r * np.cos(theta), r * np.sin(theta))))


if __name__ == '__main__':

    wound = CellsOnWoundMargin()
    c = wound.concentration(np.ones(7), np.linspace(1, 9, 9), np.linspace(1, 9, 9))
    print(c)