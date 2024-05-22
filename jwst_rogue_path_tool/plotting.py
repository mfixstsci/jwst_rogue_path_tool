"""
This module contains all of the plotting functions for the JWST Rogue Path Tool.
Currently, the tool supports exposure and observation level plots.

Authors
-------
    - Mario Gennaro
    - Mees Fix

Use
---
    Routines in this module can be imported as follows:

    >>> from jwst_rogue_path_tool.plotting import create_exposure_plots, create_observation_plots
    >>> from jwst_rogue_path_tool.detect_claws import AptProgram

    >>> filename = "/path/to/sql_apt_file.sql"
    >>> program = AptProgram(filename, angular_step=1, usr_defined_obs=[1])
    >>> program.run()

    >>> observation = program.observations.data[1]  # get obs_id 1 from program
    >>> ra, dec = program.ra, program.dec
    >>> create_exposure_plots(observation, ra, dec)
    >>> create_observation_plot(observation, ra, dec)
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from itertools import chain
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import operator
from pysiaf.utils import rotations


def create_exposure_plots(observation, ra, dec, **kwargs):
    """Generate exposure level plots

    Parameters
    ----------
    ra : float
        Right Ascension in degrees

    dec : float
        Declination in degrees

    **kwarg : dict
        Arbitrary keyword arguements
    """

    plt.rcParams["figure.figsize"] = (20, 15)

    inner_radius = kwargs.get("inner_radius", 8.0)
    outer_radius = kwargs.get("outer_radius", 12.0)
    ncols = kwargs.get("ncols", 4)

    wedge_length = inner_radius - 1
    exposure_frames = observation["exposure_frames"]
    exposure_frames_data = exposure_frames.data

    nrows = len(exposure_frames_data) // ncols + (len(exposure_frames_data) % ncols > 0)

    catalog = exposure_frames.catalog
    plotting_catalog = locate_targets_in_annulus(
        catalog, ra, dec, inner_radius, outer_radius
    )

    obs_id = observation["visit"]["observation"][0]

    for n, exp_num in enumerate(exposure_frames_data):
        angle_start = exposure_frames.valid_starts_angles[exp_num]
        angle_end = exposure_frames.valid_ends_angles[exp_num]

        # Some exposures will have no valid/consecutive angles.
        # if not angle_start or angle_end:
        #     continue

        ax = plt.subplot(nrows, ncols, n + 1)
        ax.set_xlabel("RA [Degrees]")
        ax.set_ylabel("DEC [Degrees]")
        ax.set_title("Observation {}, Exposure: {}".format(obs_id, exp_num))
        ax.scatter(ra, dec, marker="X", c="red")
        ax.scatter(
            plotting_catalog["ra"],
            plotting_catalog["dec"],
            c="deeppink",
        )

        ax.invert_xaxis()

        for angles in zip(angle_start, angle_end):
            min_theta = min(angles)
            max_theta = max(angles)
            w = Wedge(
                (ra, dec),
                wedge_length,
                min_theta,
                max_theta,
                fill=False,
                color="darkseagreen",
                joinstyle="round",
            )
            ax.add_artist(w)

        for angle in np.concatenate([angle_start, angle_end]):
            exposure_frames.calculate_attitude(angle)
            sus_region_patches = get_susceptibility_region_patch(
                exposure_frames, exp_num
            )
            for patch in sus_region_patches:
                ax.add_patch(patch)

    plt.tight_layout()
    plt.savefig("output.png")


def create_observation_plot(observation, ra, dec, **kwargs):
    """Plot that describe all valid angles at the observation level.
    The observation level plot is a single plot of all valid angles
    from a set of exposures.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees

    dec : float
        Declination in degrees

    **kwarg : dict
        Arbitrary keyword arguements
    """

    plt.rcParams["figure.figsize"] = (10, 10)

    inner_radius = kwargs.get("inner_radius", 8.0)
    outer_radius = kwargs.get("outer_radius", 12.0)

    wedge_length = inner_radius - 1
    exposure_frames = observation["exposure_frames"]
    exposure_frames_data = exposure_frames.data

    catalog = exposure_frames.catalog
    plotting_catalog = locate_targets_in_annulus(
        catalog, ra, dec, inner_radius, outer_radius
    )

    all_starting_angles = [
        exposure_frames.valid_starts_angles[exp_num] for exp_num in exposure_frames_data
    ]
    all_ending_angles = [
        exposure_frames.valid_ends_angles[exp_num] for exp_num in exposure_frames_data
    ]

    all_starting_angles = np.unique(np.concatenate(all_starting_angles))
    all_ending_angles = np.unique(np.concatenate(all_ending_angles))

    ax = plt.subplot()
    ax.set_xlabel("RA [Degrees]")
    ax.set_ylabel("DEC [Degrees]")
    ax.set_title("Observation Level Plot")
    ax.scatter(ra, dec, marker="X", c="red")
    ax.scatter(plotting_catalog["ra"], plotting_catalog["dec"], c="deeppink")

    ax.invert_xaxis()

    for angles in zip(all_starting_angles, all_ending_angles):
        min_theta = min(angles)
        max_theta = max(angles)
        w = Wedge(
            (ra, dec),
            wedge_length,
            min_theta,
            max_theta,
            fill=False,
            color="darkseagreen",
            joinstyle="round",
        )
        ax.add_artist(w)

    plt.tight_layout()
    plt.show()


def get_susceptibility_region_patch(exposure_frames, exposure_id):
    """Obtain data for susceptibility region and generate plottable
    patch.

    Parameters
    ----------
    exposure_frames : jwst_rogue_path_tool.detect_claws.ExposureFrames
        ExposureFrame object associated with observation.
    """
    patches = []

    region = exposure_frames.susceptibilty_region

    for key in region:
        module = region[key]
        attitude = exposure_frames.attitude

        v2 = 3600 * module.V2V3path.vertices.T[0]  # degrees --> arcseconds
        v3 = 3600 * module.V2V3path.vertices.T[1]  # degrees --> arcseconds

        ra_rads, dec_rads = rotations.tel_to_sky(
            attitude, v2, v3
        )  # returns ra and dec in radians
        ra_deg, dec_deg = (
            ra_rads * 180.0 / np.pi,
            dec_rads * 180.0 / np.pi,
        )  # convert to degrees

        ra_dec_path = Path(np.array([ra_deg, dec_deg]).T, module.V2V3path.codes)
        ra_dec_patch = PathPatch(ra_dec_path, lw=2, alpha=0.1)
        patches.append(ra_dec_patch)

    return patches


def locate_targets_in_annulus(catalog, ra, dec, inner_radius, outer_radius):
    """Calculate the targets from a catalog that fall within inner and outer radii.

    Parameters
    ----------
    catalog : pandas.core.frame.DataFrame
        DataFrame of star positions and magnitudes from 2MASS

    ra : float
        Right Ascension in degrees

    dec : float
        Declination in degrees

    inner_radius : float
        Inner radius of annulus

    outer_radius : float
        Outer area of annulus
    """

    # Set coordinates for target and catalog
    target_coordinates = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    catalog_coordinates = SkyCoord(
        catalog["ra"].values * u.deg,
        catalog["dec"].values * u.deg,
        frame="icrs",
    )

    # Calculate separation from target to all targets in catalog
    separation = target_coordinates.separation(catalog_coordinates)
    mask = (separation.deg < outer_radius) & (separation.deg > inner_radius)

    # Retrieve all targets in masked region above.
    plotting_catalog = catalog[mask]

    return plotting_catalog


def plot_fixed_angle_regions(observation, angle, savefig=False):
    program = observation["nircam_templates"]["program"][0]
    observation_number = observation["nircam_templates"]["observation"][0]
    susceptibility_region = observation["exposure_frames"].susceptibility_region
    number_of_modules = len(susceptibility_region)
    modules_name = observation["nircam_templates"]["modules"][0]

    plt.set_cmap("magma")
    fig, ax = plt.subplots(number_of_modules, figsize=(15, 15))

    # Hack for the loop below to work with a program that contains
    # a single module (A or B) or both modules (A and B).
    try:
        len(ax)
    except TypeError:
        ax = [ax]

    for ax, module in zip(ax, susceptibility_region):
        centroid = susceptibility_region[module].centroid
        averages = observation[f"averages_{module}"][angle]

        avg_v2 = averages[f"avg_v2_{module}"]
        avg_v3 = averages[f"avg_v3_{module}"]
        avg_intensity = averages[f"avg_intensity_{module}"]

        im = ax.scatter(avg_v2, avg_v3, c=avg_intensity)
        fig.colorbar(im, ax=ax, label="Intensity")

        # Make box around centroid of centroid of susceptibility region.
        ax.set_xlim([centroid[0] - 5, centroid[0] + 5])
        ax.set_ylim([centroid[1] - 5, centroid[1] + 5])

        ax.set_xlabel("V2")
        ax.set_ylabel("V3")
        ax.title.set_text(
            f"Program: {program} Observation: {observation_number} Angle: {angle} Module: {module}"
        )

        patch = PathPatch(susceptibility_region[module].V2V3path, alpha=0.1)
        ax.add_patch(patch)

    if savefig:
        output_filename = f"{program}_{observation_number}_{modules_name}_{angle}.png"
        plt.savefig(output_filename)
    else:
        plt.show()

    plt.close()


def plot_flux_vs_v3pa(observation):
    observation_number = observation["nircam_templates"]["observation"][0]

    flux = observation["flux_vs_angle"]

    fig, axes = plt.subplots(len(flux.keys()), figsize=(15, 10))

    fig.suptitle("Main title")

    for ax, key in zip(axes, flux.keys()):
        flux_values = flux[key]
        ax.plot(np.arange(len(flux_values)), flux_values)
        ax.set_title(key)
        ax.set_xlabel("V3PA")
        ax.set_ylabel("DN/pix/ks")

    plt.tight_layout()

    plt.show()
