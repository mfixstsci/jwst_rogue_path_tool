"""Utility funcitons for JWST Rogue Path Tool


Authors
-------
    - Mario Gennaro
    - Mees Fix
"""

from jwst_backgrounds import jbt
import numpy as np
import pandas as pd
import pathlib

from jwst_rogue_path_tool.constants import PROJECT_DIRNAME


def absolute_magnitude(band_magnitude):
    """Calculate absolute magnitude from band magnitude.

    Parameters
    ----------
    band_magnitude : float
        Band magnitude value from catalog

    Returns
    -------
    absolute_magnitude : float
        Calculated absolute magnitude of target in catalog.
    """
    absolute_magnitude = -2.5 * np.log10(band_magnitude)
    return absolute_magnitude


def calculate_background(ra, dec, wavelength, threshold):
    """Calculate background using JWST Backgrounds Tool

    Parameters
    ----------
    ra : float
        Right acension of target
    dec : float
        Declination of target
    wavelength : float
        ????
    threshold : floaf
        ????

    Returns
    -------
    background_data :
    """
    background_data = jbt.background(ra, dec, wavelength=wavelength, thresh=threshold)
    return background_data


def get_pupil_from_filter(filters):
    """Given a NRC filter, return list of available filters

    Parameters
    ----------
    filters : list
        List of NRC filter names

    Returns
    -------
    pupils : dict
        A dictionary of key (filter) and value (pupil)
    """
    pupils = {}

    for fltr in filters:
        if "+" in fltr:
            splt = fltr.split("+")
            pupil = splt[0]
            filter = splt[1]
        elif "_" in fltr:
            splt = fltr.split("_")
            pupil = splt[0]
            filter = splt[1]
        else:
            filter = fltr
            pupil = "CLEAR"

        pupils[filter] = pupil

    return pupils


def get_pivot_wavelength(pupil, filter):
    """Get pivot wavelength values from filter_data.txt

    Parameters
    ----------
    pupil : str
        NRC pupil name
    filter : str
        NRC filter name

    Returns
    -------
    pivot_wavelength : float
        Pivot wavelength of filter/pupil
    """
    filter_filename = pathlib.Path(PROJECT_DIRNAME) / "data" / "filter_data.txt"

    filter_table = pd.read_csv(filter_filename, sep="\s+")  # noqa

    if pupil == "CLEAR":
        check_value = filter
    else:
        check_value = pupil

    BM = filter_table["Filter"] == check_value
    pivot_wavelength = filter_table.loc[BM, "Pivot"].values[0]

    return pivot_wavelength


def make_output_directory(directory_name):
    """Make output directories for figures and text files.

    Parameters
    ----------
    directory_name : str
        Path to create directory, if exists, code will not fail but not overwrite.
    """
    pathlib.Path(directory_name).mkdir(parents=True, exist_ok=True)
