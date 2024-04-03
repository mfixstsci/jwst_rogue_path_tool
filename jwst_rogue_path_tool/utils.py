from functools import reduce
from itertools import groupby

import numpy as np

def get_consecutive_valid_angles(angles, step):
    """Find consecutive angle values that aren't larger than step.
    """
    window_indices = []
    try:
        window_start_index = angles[0]
    except IndexError:
        return None
    for x,y in zip(angles[::], angles[1::]):
        difference = y - x
        if difference > step:
            window_end_index = x
            if window_end_index == window_start_index:
                window_start_index = y
                continue
            else:
                window_indices.append((window_start_index, window_end_index))
                window_start_index = y
        if y == angles[-1]:
            window_indices.append((window_start_index, y))
        else:
            continue

    return window_indices

def get_intersecting_angles(angles):
    """Return intersecting angles from list of arrays of angles.
    """
    intersecting_angles = reduce(np.intersect1d, angles)

    return intersecting_angles
