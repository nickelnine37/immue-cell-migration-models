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

class Wound(PointSource):

    def __init__(self, position: np.array, units='microns'):
        super().__init__(position, units)

    def concentration(self, params, r, t):
        raise NotImplementedError

class PointWound(Wound):

    def __init__(self, position: np.array=np.array([0, 0])):
        super().__init__(position)

    def concentration(self, params, r, t):
        return concentration(params, r, t)

class MultiPointWound(Wound):

    def __init__(self, centre: np.ndarray, n_cells: int, radius: float):
        """
        The base class for a wound with multimple cells emmitting attractant

        Parameters
        ----------
        centre      The centre position of the wound
        n_cells     The number of cells emmitting attractant in the wound
        radius      The radius of the wound

        """
        super().__init__(centre)
        self.n_cells = n_cells
        self.radius = radius

    def concentration(self, params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.ndarray, float]):
        """
        Return the concentration of attractant at a distance r from the wound centre at time t.
        Note that, since a wound with multiple cells inside will not emmit attractant in a
        cicularly symmetric fashion, this is only an approximation. Really, attractant concentration
        will vary with angle theta too. The approximation is made by simply taking r as the
        y-coordinate: C(r) ≈ C(x=0, y=r)

        Notes:
            * if a single point and single time is passed, a single concentration is returned
            * if an array of points and a single time is passed, the conentration at each
              point, at that time is returned
            * if an array of points and an array of times is passed, then, if they have
              the same shape, the concentration at [(r1, t1), (r2, t2) ... ] is returned.
              But if they have different shapes, the concentration at each point at each
              time is returned
            * if a single point at an array of times is passed, an array containing the
              concentration for that point at each time is returned

        Parameters
        ----------
        params      A numpy array containing q, D and tau
        r           The distance. Can be a float or a numpy array for a sequence of positions
        t           The time. Can also be a float or a nnumpy array for a sequence of times.

        Returns
        -------

        the concentration of attractant at distance r, time t

        """

        # we want the concentration at a single point r and single time t
        if isinstance(r, (float, int)) and isinstance(t, (float, int)):
            return concentration(params, (self.x ** 2 + (r - self.y) ** 2) ** 0.5, t).sum()

        # we want the concentration at a sequence of points r and a single time t
        elif isinstance(r, np.ndarray) and isinstance(t, (float, int)):
            rs = np.concatenate([np.expand_dims((self.x[i] ** 2 + (r - self.y[i]) ** 2), axis=-1) for i in range(self.n_cells)], axis=-1)
            return concentration(params, rs, t).sum(-1)

        # we want the concentration at a sequence of points r and sequence of times t
        elif isinstance(r, np.ndarray) and isinstance(t, np.ndarray):

            # assume we want C = [C(r1, t1), C(r2, t2) ... ]
            if r.shape == t.shape:
                rs = np.concatenate([np.expand_dims((self.x[i] ** 2 + (r - self.y[i]) ** 2), axis=-1) for i in range(self.n_cells)], axis=-1)
                ts = np.concatenate([np.expand_dims(t, axis=-1) for _ in range(self.n_cells)], axis=-1)
                return concentration(params, rs, ts).sum(-1)

            # assume we want all points r at all times t
            else:
                # this checks if t is 1D
                if sum(t.shape) - t.ndim + 1 == len(t.reshape(-1)):
                    t = t.reshape(-1)
                    rs = np.concatenate([np.expand_dims((self.x[i] ** 2 + (r - self.y[i]) ** 2), axis=-1) for i in range(self.n_cells)], axis=-1)
                    ts = np.concatenate([np.expand_dims(ti * np.ones_like(rs), axis=-1) for ti in t], axis=-1)
                    rs = np.concatenate([np.expand_dims(rs, axis=-1) for _ in range(len(t))], axis=-1)
                    return concentration(params, rs, ts).sum(-2)
                else:
                    raise ValueError('If you want concentration at all points r at all times t, t must be 1D')

        # we want the concentration at a single point r at a sequence of times t
        elif isinstance(r, (float, int)) and isinstance(t, np.ndarray):

            # this checks is t is 1D
            if sum(t.shape) - t.ndim + 1 == len(t.reshape(-1)):
                ts = t.reshape(1, -1)
                rs = np.concatenate([(self.x ** 2 + (r - self.y) ** 2) ** 0.5 for _ in range(ts.shape[1])], axis=1)
                ts = np.concatenate([ts for _ in range(self.x.shape[0])], axis=0)
                return concentration(params, rs, ts).sum(0).reshape(t.shape)
            else:
                raise ValueError('If you want concentration at point r at all times t, t must be 1D')

        else:
            raise ValueError('r and t must be numbers or numpy arrays')

    def concentration_xy(self, params, x, y, t):
        """
        Return the concentration of attractant at a point x-y at time t.

        Notes:
            * if a single point and single time is passed, a single concentration is returned
            * if an array of points and a single time is passed, the conentration at each
              point, at that time is returned
            * if an array of points and an array of times is passed, then, if they have
              the same shape, the concentration at [(x1, y1, t1), (x2, y2, t2) ... ] is returned.
              But if they have different shapes, the concentration at each point at each
              time is returned
            * if a single point at an array of times is passed, an array containing the
              concentration for that point at each time is returned

        Parameters
        ----------
        params      A numpy array containing q, D and tau
        x           The x coordinate(s). Can be a float or a numpy array
        y           The y coordinate(s). Can be a float or a numpy array
        t           The time. Can also be a float or a nnumpy array for a sequence of times.

        Returns
        -------

        the concentration of attractant at point xy at time t

        """

        # x and y must be the same type
        if type(x) != type(y):
            raise ValueError(
                'x and y must be the same type, but they are {} and {} respectively'.format(type(x), type(y)))

        # we want the concentration at a single point xy at a single time t
        if isinstance(x, (float, int)) and isinstance(t, (float, int)):
            rs = ((x - self.x) ** 2 + (y - self.y ** 2)) ** 0.5
            return concentration(params, rs, t).sum()

        # we want the concentration at a sequence of points xy at a single time t
        if isinstance(x, np.ndarray) and isinstance(t, (float, int)):

            # x and y must have the same shape
            if x.shape != y.shape:
                raise ValueError(
                    'x and y must be the same shape, but they are {} and {} respectively'.format(x.shape, y.shape))

            rs = np.concatenate(
                [np.expand_dims(((x - self.x[i]) ** 2 + (y - self.y[i]) ** 2), axis=-1) for i in range(self.n_cells)],
                axis=-1)
            return concentration(params, rs, t).sum(-1)

        elif isinstance(x, np.ndarray) and isinstance(t, np.ndarray):

            # x and y must have the same shape
            if x.shape != y.shape:
                raise ValueError(
                    'x and y must be the same shape, but they are {} and {} respectively'.format(x.shape, y.shape))

            # assume we want C = [C(r1, t1), C(r2, t2) ... ]
            if x.shape == t.shape:
                rs = np.concatenate([np.expand_dims(((x - self.x[i]) ** 2 + (y - self.y[i]) ** 2), axis=-1) for i in
                                     range(self.n_cells)], axis=-1)
                ts = np.concatenate([np.expand_dims(t, axis=-1) for _ in range(self.n_cells)], axis=-1)
                return concentration(params, rs, ts).sum(-1)

            # assume we want all points r at all times t
            else:
                # this checks if t is 1D
                if sum(t.shape) - t.ndim + 1 == len(t.reshape(-1)):
                    t = t.reshape(-1)
                    rs = np.concatenate([np.expand_dims(((x - self.x[i]) ** 2 + (y - self.y[i]) ** 2), axis=-1) for i in
                                         range(self.n_cells)], axis=-1)
                    ts = np.concatenate([np.expand_dims(ti * np.ones_like(rs), axis=-1) for ti in t], axis=-1)
                    rs = np.concatenate([np.expand_dims(rs, axis=-1) for _ in range(len(t))], axis=-1)
                    return concentration(params, rs, ts).sum(-2)
                else:
                    raise ValueError('If you want concentration at all points r at all times t, t must be 1D')

        # we want the concentration at a single point r at a sequence of times t
        elif isinstance(x, (float, int)) and isinstance(t, np.ndarray):

            # this checks is t is 1D
            if sum(t.shape) - t.ndim + 1 == len(t.reshape(-1)):
                ts = t.reshape(1, -1)
                rs = np.concatenate([((x - self.x ** 2) + (y - self.y) ** 2) ** 0.5 for _ in range(ts.shape[1])],
                                    axis=1)
                ts = np.concatenate([ts for _ in range(self.x.shape[0])], axis=0)
                return concentration(params, rs, ts).sum(0).reshape(t.shape)

            else:

                raise ValueError('If you want concentration at point r at all times t, t must be 1D')

        else:
            raise ValueError('x, y and t must be numbers or numpy arrays')

class CellsOnWoundMargin(MultiPointWound):

    def __init__(self, centre: np.array=np.array([0, 0]), n_cells=10, radius=15):
        super().__init__(centre, n_cells, radius)
        self.positions = centre + radius * self.circle(n_cells)
        self.x = self.positions[:, 0].reshape(-1, 1)
        self.y = self.positions[:, 1].reshape(-1, 1)

    def circle(self, n_cells):
        """
        Spread n_cells out evenly on a unit circle

        Parameters
        ----------
        n_cells     The number of cells

        Returns
        -------
        A (n_cells, 2) numpy array containing the coordinates

        """

        return np.array([(np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, n_cells + 1)[:-1]])


class CellsInsideWound(MultiPointWound):

    def __init__(self, centre: np.array=np.array([0, 0]), n_cells=9, radius=15):
        super().__init__(centre, n_cells, radius)
        self.positions = centre + radius * self.sunflower(n_cells)
        self.x = self.positions[:, 0].reshape(-1, 1)
        self.y = self.positions[:, 1].reshape(-1, 1)

    def sunflower(self, n):
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


def concentration(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[np.ndarray, float]:
    """
    This function returns the concentration of attractant at a
    radial distance r and time t for a continuous point source
    emitting attractand at a rate q from the origin from t=0
    to t=tau with diffusion constant D.

    Parameters
    ----------
    params  A numpy array containing q, D and tau
    r       the radial distance from the origin. Can be float or array
    t       time

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





if __name__ == '__main__':

    wound = CellsOnWoundMargin()
    c = wound.concentration(np.ones(7), np.linspace(1, 9, 9), np.linspace(1, 9, 9))
    print(c)