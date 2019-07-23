import numpy as np
from utils.checks import check_is_numpy

def nan_concatenate(arrays: list, axis: int):
    """
    A function similar to numpy concatenate, except if the arrays are of
    different length along the non-concatenating axis, fill them with nans.
    Only works for

    e.g.
    > a = np.zeros((3, 3))
    > b = np.zeros((2, 3))
    > c = nan_concatenate([a, b], axis=1)

    c = [0,   0,   0,  0,   0,   0]
        [0,   0,   0,  0,   0,   0]
        [0,   0,   0, nan, nan, nan]


    Parameters
    ----------
    arrays
    axis

    Returns
    -------

    """

    assert isinstance(arrays, list)

    for array in arrays:
        check_is_numpy(array)

    shapes = [array.shape for array in arrays]
    assert all([len(shape) == len(shapes[0]) for shape in shapes]), 'All arrays passed must have the same number of dimensions'
    shapes = np.array(shapes)
    max_dims = shapes.max(0)

    new_arrays = []
    for array in arrays:
        new_shape = max_dims.copy()
        new_shape[axis] = array.shape[axis]
        new_array = np.zeros(new_shape)
        new_array[:] = np.nan
        slices = tuple([slice(0, dim_len) for dim_len in array.shape])
        new_array[slices] = array
        new_arrays.append(new_array)

    return np.concatenate(new_arrays, axis=axis)


if __name__ == '__main__':

    a = np.zeros((3, 3))
    b = np.zeros((2, 3))
    c = nan_concatenate([a, b], axis=1)
    print(c)