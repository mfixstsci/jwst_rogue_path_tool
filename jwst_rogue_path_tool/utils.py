from jwst_backgrounds import jbt
import numpy as np
import os
import pandas as pd

from jwst_rogue_path_tool.constants import PROJECT_DIRNAME


def absolute_magnitude(band_magnitude):
    absolute_magnitude = -2.5 * np.log10(band_magnitude)
    return absolute_magnitude


def calculate_background(ra, dec, wavelength, threshold):
    background_data = jbt.background(ra, dec, wavelength=wavelength, thresh=threshold)
    return background_data


def get_pupil_from_filter(filters):
    pupils = {}

    for fltr in filters:
        if "+" in fltr:
            splt = filter.split("+")
            pupil = splt[0]
            filter = splt[1]
        elif "_" in fltr:
            splt = filter.split("_")
            pupil = splt[0]
            filter = splt[1]
        else:
            pupil = "CLEAR"
            filter = fltr

        pupils[filter] = pupil

    return pupils


def get_pivot_wavelength(pupilshort, filtershort):
    filter_filename = os.path.join(PROJECT_DIRNAME, "data", "filter_data.txt")

    filter_table = pd.read_csv(filter_filename, sep="\s+")

    if pupilshort == "CLEAR":
        check_value = filtershort
    else:
        check_value = pupilshort

    BM = filter_table["Filter"] == check_value

    return filter_table.loc[BM, "Pivot"].values[0]
