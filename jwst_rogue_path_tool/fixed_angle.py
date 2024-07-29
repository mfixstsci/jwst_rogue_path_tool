"""
This module contains the FixedAngle class which performs the analysis for
JWST Rogue Path Tool.

This takes an observation and converts the magnitudes of stars from the catalog
and converts them into flux.

Authors
-------
    - Mario Gennaro
    - Mees Fix

Use
---
    Routines in this module can be imported as follows:

    >>> from jwst_rogue_path_tool.fixed_angle import FixedAngle
    >>> fa = FixedAngle(observation, 259.0)
    >>> fa.plot_regions()
    >>> fa.plot_flux_vs_v3pa()
"""

import operator

from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np

from jwst_rogue_path_tool.constants import (
    CATALOG_BANDPASS,
    NIRCAM_ZEROPOINTS,
    ZEROPOINT,
)

from jwst_rogue_path_tool.plotting import plot_fixed_angle_regions
from jwst_rogue_path_tool.utils import absolute_magnitude, get_pupil_from_filter

np.seterr(divide="ignore")  # For np.log10 divide by 0


class FixedAngle:
    def __init__(self, observation, angle):
        self.observation = observation
        self.angle = angle
        self.catalog = self.observation["exposure_frames"].catalog
        self.catalog_name = self.observation["exposure_frames"].catalog_name
        self.bands = CATALOG_BANDPASS[self.catalog_name]

        self.exposure_frame_table = self.observation[
            "exposure_frames"
        ].exposure_frame_table

        self.susceptibility_region = self.observation[
            "exposure_frames"
        ].susceptibility_region

        self.filter_modules_combos = self.exposure_frame_table.groupby(
            ["filter_short", "modules"]
        ).size()

        self.total_exposure_duration_table = self.observation[
            "exposure_frames"
        ].total_exposure_duration_table

        self.filters = self.total_exposure_duration_table.index.values
        self.pupils = get_pupil_from_filter(self.filters)
        self.calculate_absolute_magnitude()
        self.get_total_magnitudes()
        self.get_total_counts()

    def calculate_absolute_magnitude(self):
        for module in self.susceptibility_region:
            for band in self.bands:
                average_intensity = self.observation[f"averages_{module}"][self.angle][
                    f"avg_intensity_{module}"
                ]
                abs_magnitude = (
                    self.catalog[band]
                    + absolute_magnitude(average_intensity)
                    + ZEROPOINT
                )

                # Test and see if this can be removed, using np.inf should be fine.
                abs_magnitude.replace(-np.inf, np.nan, inplace=True)

                self.catalog[f"abs_mag_{band}_{module}"] = abs_magnitude

    def get_empirical_zero_points(self, module, pupil, filter):
        if module == "A":
            return NIRCAM_ZEROPOINTS["zeropoints_A"][f"{pupil}+{filter}"]
        elif module == "B":
            return NIRCAM_ZEROPOINTS["zeropoints_B"][f"{pupil}+{filter}"]
        else:
            assert ValueError(f"module must be value 'A' or 'B' not {module}")

    def get_ground_band(self, pupil, filter, catalog="2MASS"):
        if catalog == "2MASS":
            return NIRCAM_ZEROPOINTS["match2MASS"][f"{pupil}+{filter}"]
        elif catalog == "SIMBAD":
            return NIRCAM_ZEROPOINTS["matchSIMBAD"][f"{pupil}+{filter}"]
        else:
            assert ValueError(
                f"Catalog must be value '2MASS' or 'SIMBAD' not {catalog}"
            )

    def get_total_counts(self):
        self.total_counts = {}
        for module in self.susceptibility_region:
            for filter in self.filters:
                # Obtain ground based magnitude based on filter/pupil combo.
                pupil = self.pupils[filter]
                ground_band = self.get_ground_band(pupil, filter)
                # Obtain emperical (Vega Mag) zeropoint based on filter/pupil combo
                emperical_zeropoint = self.get_empirical_zero_points(
                    module, self.pupils[filter], filter
                )

                # Get summed abs magnitude for entire band.
                summed_abs_mag = self.total_magnitudes[
                    f"total_mag_{ground_band}_{module}"
                ]

                total_exposure_duration = self.total_exposure_duration_table[filter]
                tot_cnts = (
                    10 ** ((emperical_zeropoint - summed_abs_mag) / 2.5)
                    * total_exposure_duration
                )

                self.total_counts[f"total_counts_{pupil}+{filter}_{module}"] = tot_cnts

    def get_total_magnitudes(self):
        self.total_magnitudes = {}
        for module in self.susceptibility_region:
            for band in self.bands:
                column_name = f"abs_mag_{band}_{module}"
                if column_name in self.catalog.columns:
                    summed_magnitude = np.nansum(
                        np.power(10, -0.4 * self.catalog[column_name].values)
                    )
                    total_mag = absolute_magnitude(summed_magnitude)
                else:
                    raise Exception(f"{column_name} not available in catalog!")
                self.total_magnitudes[f"total_mag_{band}_{module}"] = total_mag

    def plot_regions(self):
        """Convenience method for plotting susceptibility region for a single angle
        in V2/V3 space with maginitude intensity."""
        plot_fixed_angle_regions(self.observation, self.angle)
