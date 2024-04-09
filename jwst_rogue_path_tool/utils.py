from functools import reduce
from itertools import groupby

import numpy as np


def get_valid_angles_windows(angles):
    change = np.where(angles[:-1] != angles[1:])[0]
    if change.size >0:
        if angles[change[0]]:
            change = np.roll(change,1)
            window_start = angles[change[::2]]
            window_end = angles[change[1::2]]

    return window_start, window_end
