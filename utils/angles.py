import numpy as np

def angle_between(v1, v2):
    """
    v1 is the reference vector. Angle is positive anti-clockwise, negative clockwise
    between -pi and pi.
    """
    dot = v1 @ v2
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(det, dot)
