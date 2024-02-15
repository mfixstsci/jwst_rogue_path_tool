from functools import reduce
from itertools import groupby

import numpy as np

def get_consecutive_valid_angles(angles):
    # Enumerate and get differences between counter—integer pairs
    # Group by differences (consecutive integers have equal differences)  
    gb = groupby(enumerate(angles), key=lambda x: x[0] - x[1])

    # Repack elements from each group into list
    all_groups = ([i[1] for i in g] for _, g in gb)

    # Filter out one element lists
    consecutive_angles = list(filter(lambda x: len(x) > 1, all_groups))

    return consecutive_angles

def get_intersecting_angles(angles):
    """Return intersecting angles from list of arrays of angles.
    """
    intersecting_angles = reduce(np.intersect1d, angles)

    return intersecting_angles

