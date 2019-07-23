import numpy as np


def check_is_numpy(obj, name: str=''):
    """
    Check is an object is a numpy array

    Parameters
    ----------
    obj        The object
    name       The variable name

    Returns
    -------
    True       If obj is a numpy array. Else raise TypeError
    """

    if isinstance(obj, np.ndarray):
        return True
    else:
        raise TypeError('{} should be a numpy aray, but it is {}'.format(name, type(obj)))

def check_valid_prior(priors):
    """
    Check whether a prior, or list or priors are valid for use in MCMC walks

    Parameters
    ----------
    priors      A prior or a list of prior (from utils.distributions)

    Returns
    -------
    True        If all priors are valid. Else ValueError is raised

    """

    if isinstance(priors, list):

        if all([hasattr(prior, 'valid_prior') for prior in priors]):
            if all([prior.valid_prior for prior in priors]):
                return True
            else:
                raise ValueError('One or more of the priors passed are invalid')

        else:
            raise ValueError('One or more of the priors passed are invalid')

    else:
        if hasattr(priors, 'valid_prior'):
            if priors.valid_prior:
                return True
            else:
                raise ValueError('The prior passed is invalid')
        else:
            raise ValueError('The prior passed in invalid')

def assert_same_length(*args):

    l1 = len(args[0])
    for arg in args:
        if len(arg) != l1:
            raise ValueError('These objects must be the same length')

    return True
