import json
import os

from astroquery.simbad import Simbad
from matplotlib.path import Path


__location__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


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


class SusceptibilityZoneVertices:
    """Convenience class that creates a matplotlib.Path with the Rogue Path
    susceptibility zone vertices for a given NIRCam module"""

    def __init__(self, module="A", small=False):
        self.small = small
        self.module = module
        self.V2V3path = self.get_path()

    def get_path(self):
        if self.module == "A":
            if self.small == False:
                V2list = [
                    2.64057,
                    2.31386,
                    0.47891,
                    0.22949,
                    -0.04765,
                    -0.97993,
                    -0.54959,
                    0.39577,
                    0.39577,
                    1.08903,
                    1.56903,
                    2.62672,
                    2.64057,
                ]
                V3list = [
                    10.33689,
                    10.62035,
                    10.64102,
                    10.36454,
                    10.65485,
                    10.63687,
                    9.89380,
                    9.47981,
                    9.96365,
                    9.71216,
                    9.31586,
                    9.93600,
                    10.33689,
                ]
            else:
                V2list = [
                    2.28483,
                    0.69605,
                    0.43254,
                    0.57463,
                    0.89239,
                    1.02414,
                    1.70874,
                    2.28483,
                    2.28483,
                ]
                V3list = [
                    10.48440,
                    10.48183,
                    10.25245,
                    10.12101,
                    10.07204,
                    9.95349,
                    10.03854,
                    10.04369,
                    10.48440,
                ]

        else:
            if self.small == False:
                V2list = [
                    0.52048,
                    0.03549,
                    -0.28321,
                    -0.49107,
                    -2.80515,
                    -2.83287,
                    -1.58575,
                    -0.51878,
                    -0.51878,
                    -0.40792,
                    0.11863,
                    0.70062,
                    0.52048,
                ]
                V3list = [
                    10.32307,
                    10.32307,
                    10.01894,
                    10.33689,
                    10.33689,
                    9.67334,
                    9.07891,
                    9.63187,
                    8.99597,
                    8.96832,
                    9.21715,
                    9.70099,
                    10.32307,
                ]
            else:
                V2list = [
                    -0.96179,
                    -1.10382,
                    -2.41445,
                    -2.54651,
                    -2.54153,
                    -2.28987,
                    -1.69435,
                    -1.46262,
                    -1.11130,
                    -0.95681,
                    -0.59551,
                    -0.58306,
                    -0.96179,
                ]
                V3list = [
                    10.03871,
                    10.15554,
                    10.15554,
                    10.04368,
                    9.90945,
                    9.82741,
                    9.76030,
                    9.64347,
                    9.62855,
                    9.77273,
                    9.88459,
                    10.07848,
                    10.03871,
                ]

        V2list = [-1.0 * v for v in V2list]

        verts = []
        for xx, yy in zip(V2list, V3list):
            verts.append((xx, yy))

        codes = [Path.MOVETO]

        for _ in verts[1:-1]:
            codes.append(Path.LINETO)

        codes.append(Path.CLOSEPOLY)

        return Path(verts, codes)
