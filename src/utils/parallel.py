import joblib
from typing import Union

def parallel_methods(objects: list,
                     method_names: Union[str, list],
                     params: Union[dict, list]=None,
                     n_jobs: int=-1,
                     backend="multiprocessing") -> list:
    """
    Parallelise the execution of a set of methods on a set of objects.

    Params:
        objects:       list of objects, which will have their methods executed in parallel
        method_names:  str of method name, can be list if diff for each obj
        params:        kwargs for method. list of dicts if each obj different, otherwise single dict if all the same
        n_jobs:        -1 for max, else int
        backend:       "multiprocessing" or "threading"

    Returns:
        list of the return values for each object method call

    """

    if params is None:
        params = {}
    if isinstance(params, dict):
        params = [params] * len(objects)
    assert len(objects) == len(params), "params list should have same length as number of objects"

    if isinstance(method_names, str):
        method_names = [method_names] * len(objects)
    assert len(objects) == len(method_names), "methods list should have same length as number of objects"

    res = joblib.Parallel(n_jobs=n_jobs, backend=backend)(joblib.delayed(obj.__getattribute__(mn))(**param) for obj, mn, param in zip(objects, method_names, params))
    return res


def parallel_functions(funcs: list,
                       params: list,
                       n_jobs: int=-1,
                       backend="multiprocessing"):
    """
    Parallelise the execution of a set of functions.

    Params:
        funcs:       list of functions, which will be executed in parallel
        params:      kwargs for method. list of dicts if each obj different, otherwise single dict if all the same
        n_jobs:      -1 for max, else int
        backend:     "multiprocessing" or "threading"

    Returns:
        list of the return values for each function call
    """

    if params is None:
        params = {}
    if isinstance(params, dict):
        params = [params] * len(funcs)
    assert len(funcs) == len(params), "params list should have same length as number of objects"

    res = joblib.Parallel(n_jobs=n_jobs, backend=backend)(joblib.delayed(func)(**param) for func, param in zip(funcs, params))

    return res

