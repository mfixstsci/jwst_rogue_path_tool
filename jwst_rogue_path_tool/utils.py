import numpy as np

def absolute_magnitude(band_magnitude):
    absolute_magnitude = -2.5 * np.log10(band_magnitude)
    return absolute_magnitude