import json
import os

from astroquery.simbad import Simbad
import numpy as np
import pandas as pd


__location__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def compute_line(startx, starty, angle, length):
    anglerad = np.pi / 180.0 * angle
    endx = startx + length * np.cos(anglerad)
    endy = starty + length * np.sin(anglerad)

    return np.array([startx, endx]), np.array([starty, endy])


def get_config():
    """Return a dictionary that holds the contents of the ``jwql``
    config file.

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """
    config_file_location = os.path.join(
        __location__, "jwst_rogue_path_tool", "config.json"
    )

    # Make sure the file exists
    if not os.path.isfile(config_file_location):
        raise FileNotFoundError(
            "The jwst_rogue_path_tool requires a configuration file "
            "to be placed within the main directory. "
            "This file is missing."
        )

    with open(config_file_location, "r") as config_file_object:
        try:
            # Load it with JSON
            settings = json.load(config_file_object)
        except json.JSONDecodeError as e:
            # Raise a more helpful error if there is a formatting problem
            raise ValueError(
                "Incorrectly formatted config.json file. "
                "Please fix JSON formatting: {}".format(e)
            )

    return settings


def querysimbad(ra, dec, rad=1, band="K", maxmag=6.0, simbad_timeout=200):
    """Function to put together a "query by criteria" SIMBAD query
    and return an astropy Table with the results.
    Query criteria here are a circle radius and a faint magnitude limit
    based on a user-selectable bandpass
    """

    Simbad.TIMEOUT = simbad_timeout
    Simbad.reset_votable_fields()

    for filtername in ["J", "H", "K"]:
        for prop in [
            "",
            "_bibcode",
            "_error",
            "_name",
            "_qual",
            "_system",
            "_unit",
            "data",
        ]:
            field = "flux{}({})".format(prop, filtername)
            Simbad.add_votable_fields(field)

    if ra >= 0.0:
        ra_symbol = "+"
    else:
        ra_symbol = "-"

    if dec >= 0.0:
        dec_symbol = "+"
    else:
        dec_symbol = "-"

    crit = "region(circle, ICRS, {}{} {}{},{}d) & ({}mag < {})".format(
        ra_symbol, ra, dec_symbol, dec, rad, band, maxmag
    )
    print(crit)
    t = Simbad.query_criteria(crit)

    return t
