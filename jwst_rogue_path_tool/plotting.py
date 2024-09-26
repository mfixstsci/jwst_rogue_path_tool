"""
This module contains all of the plotting functions for the JWST Rogue Path Tool.
Plotting output include:
    * Valid angles at the exposure level
    * Valid angles at the observation level
    * Susceptibility region footprint w source intensities.
    * V3 position angles vs flux at the observation level.

Authors
-------
    - Mario Gennaro
    - Mees Fix
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from pysiaf.utils import rotations

from jwst_rogue_path_tool.utils import make_output_directory


def create_exposure_plots(observation, ra, dec, output_directory=None, **kwargs):
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

    obs_id = observation["visit"]["observation"].values[0]
    program = observation["nircam_templates"]["program"].values[0]

    for n, exp_num in enumerate(exposure_frames_data):
        angle_start = exposure_frames.valid_starts_angles[exp_num]
        angle_end = exposure_frames.valid_ends_angles[exp_num]

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
        ax.axis("equal")
        ax.invert_xaxis()

        if angle_start is None and angle_end is None:
            ax.annotate("NO VALID ANGLES", xy=(ra, dec))
        else:
            for angles in zip(angle_start, angle_end):
                min_theta = (
                    90.0 - angles[0]
                )  # Origin of matplotlib is offset by 90 degrees
                max_theta = (
                    90.0 - angles[1]
                )  # Origin of matplotlib is offset by 90 degrees
                w = Wedge(
                    (ra, dec),
                    wedge_length,
                    max_theta,
                    min_theta,
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

    if output_directory:
        filename = f"exposure_plot_program_{program}_obs_id_{obs_id}.png"
        full_path = output_directory / str(program) / "exposure_level"
        make_output_directory(full_path)
        print(f"WRITING FIGURE TO {full_path / filename}")
        plt.savefig(full_path / filename)
    else:
        plt.show()

    plt.close()


def create_observation_plot(observation, ra, dec, output_directory=None, **kwargs):
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

    observation_number = observation["nircam_templates"]["observation"].values[0]
    program = observation["nircam_templates"]["program"].values[0]

    catalog = exposure_frames.catalog
    plotting_catalog = locate_targets_in_annulus(
        catalog, ra, dec, inner_radius, outer_radius
    )

    ax = plt.subplot()
    ax.set_xlabel("RA [Degrees]")
    ax.set_ylabel("DEC [Degrees]")
    ax.set_title(f"Program {program} Observation {observation_number}")
    ax.scatter(ra, dec, marker="X", c="red")
    ax.scatter(plotting_catalog["ra"], plotting_catalog["dec"], c="deeppink")

    ax.axis("equal")
    ax.invert_xaxis()

    all_starting_angles = observation["valid_starts_angles"]
    all_ending_angles = observation["valid_ends_angles"]

    if all_starting_angles.size == 0 and all_ending_angles.size == 0:
        ax.annotate("NO VALID ANGLES", xy=(ra, dec))
    else:
        for angles in zip(all_starting_angles, all_ending_angles):
            min_theta = 90.0 - angles[0]  # Origin of matplotlib is offset by 90 degrees
            max_theta = 90.0 - angles[1]  # Origin of matplotlib is offset by 90 degrees
            w = Wedge(
                (ra, dec),
                wedge_length,
                max_theta,
                min_theta,
                fill=False,
                color="darkseagreen",
                joinstyle="round",
            )
            ax.add_artist(w)

    plt.tight_layout()
    if output_directory:
        filename = f"observation_plot_program_{program}_obs_id_{observation_number}.png"
        full_path = output_directory / str(program) / "observation_level"
        make_output_directory(full_path)
        print(f"WRITING FIGURE TO {full_path / filename}")
        plt.savefig(full_path / filename)
    else:
        plt.show()

    plt.close()


def get_susceptibility_region_patch(exposure_frames, exposure_id):
    """Obtain data for susceptibility region and generate plottable
    patch.

    Parameters
    ----------
    exposure_frames : jwst_rogue_path_tool.detect_claws.ExposureFrames
        ExposureFrame object associated with observation.
    """
    patches = []

    region = exposure_frames.susceptibility_region

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


def plot_fixed_angle_regions(observation, angle, output_directory=None):
    """Plot the susceptibility region and targets in and around it.

    Parameters
    ----------
    observation : dictionary
        Dictionary of a single observation dataset

    angle : float
        Angle of attitude.

    output_directory : str
        Directory to save figure to.
    """
    program = observation["nircam_templates"]["program"].values[0]
    observation_number = observation["nircam_templates"]["observation"].values[0]
    susceptibility_region = observation["exposure_frames"].susceptibility_region
    number_of_modules = len(susceptibility_region)
    modules_name = observation["nircam_templates"]["modules"].values[0]

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
        ax.set_xlim([centroid[0] - 3, centroid[0] + 3])
        ax.set_ylim([centroid[1] - 3, centroid[1] + 3])

        ax.set_xlabel("V2")
        ax.set_ylabel("V3")
        ax.title.set_text(
            f"Program: {program} Observation: {observation_number} Angle: {angle} Module: {module}"
        )

        patch = PathPatch(susceptibility_region[module].V2V3path, alpha=0.1)
        ax.add_patch(patch)

    if output_directory:
        full_path = output_directory / str(program) / "susceptibility_regions"
        filename = f"{program}_{observation_number}_{modules_name}_{angle}.png"
        make_output_directory(full_path)
        print(f"WRITING FIGURE TO {full_path / filename}")
        plt.savefig(full_path / filename)
    else:
        plt.show(fig)

    plt.close(fig)


def create_v3pa_vs_flux_plot(observation, output_directory=None, fontsize=15):
    """Create plot of V3 position angles and flux for a single observation.

    Parameters
    ----------
    observation : dictionary
        Dictionary of data associated with a single observation.

    output_dictionary : str
        Directory to save figure to

    fontsize : int
        Fontsize for plot
    """
    observation_number = observation["nircam_templates"]["observation"].values[0]
    program_id = observation["visit"]["program"].values[0]
    susceptibility_regions = observation["exposure_frames"].susceptibility_region
    modules = susceptibility_regions.keys()
    filters = observation["filters"]
    pupils = observation["pupils"]

    flux = observation["flux"]["dn_pix_ks"]
    flux_boolean = observation["flux_boolean"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        len(filters),
        len(modules),
        figsize=(15, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Assign module name to columns of plots.
    column_names = [f"Module: {module}" for module in modules]
    for ax, col in zip(axes[0], column_names):
        ax.set_title(col, fontsize=fontsize)

    fig.suptitle(
        f"Program: {program_id} Observation: {observation_number}", fontsize=fontsize
    )

    for mod, module in enumerate(modules):
        for fltr, (filter, pupil) in enumerate(zip(filters, pupils)):
            flux_key = f"dn_pix_ks_{pupil}+{filter}_{module}"
            flux_values = flux[flux_key]
            axes[fltr, mod].plot(flux_values)
            above_threshold = np.copy(flux_values)

            for color_idx, key in enumerate(flux_boolean[f"{filter}_{module}"].keys()):
                stats_function, lam_threshold, bkg_threshold = key.split("_")
                above_threshold[
                    flux_boolean[f"flux_boolean_{stats_function}_{module}"]
                ] = np.nan
                label_str = f"{eval(bkg_threshold):.1f} x {stats_function} = {eval(lam_threshold):.1f} DN/pix/ks"
                axes[fltr, mod].plot(above_threshold, c=colors[color_idx + 1])
                axes[fltr, mod].axhline(
                    eval(lam_threshold),
                    c=colors[color_idx + 1],
                    ls="--",
                    label=label_str,
                )

            axes[fltr, mod].set_yscale("log")
            axes[fltr, mod].set_xlabel("V3PA", fontsize=fontsize)
            axes[fltr, mod].set_ylabel(f"DN/pix/ks ({filter})", fontsize=fontsize)
            axes[fltr, mod].set_ylim(0.005, 500)
            axes[fltr, mod].legend(
                loc="lower right", fontsize=fontsize - (fontsize / 4)
            )

    axes[0, 0].set_xlim(0, 360)
    fig.tight_layout()

    if output_directory:
        if len(modules) > 1:
            filename = f"v3pa_vs_flux_{program_id}_{observation_number}_ALL.png"
        else:
            filename = f"v3pa_vs_flux_{program_id}_{observation_number}_{module}.png"

        full_path = output_directory / str(program_id) / "v3pa_vs_flux"
        make_output_directory(full_path)
        print(f"WRITING FIGURE TO {full_path / filename}")
        plt.savefig(full_path / filename)
    else:
        plt.show(fig)

    plt.close(fig)
