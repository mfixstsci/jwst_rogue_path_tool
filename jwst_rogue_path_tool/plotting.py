import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from pysiaf.utils import rotations


def create_exposure_plots(observation, ra, dec, **kwargs):
    """Generate exposure level plots"""

    plt.rcParams["figure.figsize"] = (20,15)

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
            sus_region_patches = get_susceptibility_region_patch(exposure_frames, exp_num)
            for patch in sus_region_patches:
                ax.add_patch(patch)

    plt.tight_layout()
    plt.savefig('output.png')


def create_observation_plot(observation, ra, dec, **kwargs):
    """Plot that describe all valid angles at the observation level"""

    plt.rcParams["figure.figsize"] = (10,10)

    inner_radius = kwargs.get("inner_radius", 8.0)
    outer_radius = kwargs.get("outer_radius", 12.0)

    wedge_length = inner_radius - 1
    exposure_frames = observation["exposure_frames"]
    exposure_frames_data = exposure_frames.data

    catalog = exposure_frames.catalog
    plotting_catalog = locate_targets_in_annulus(
        catalog, ra, dec, inner_radius, outer_radius
    )

    all_starting_angles = [exposure_frames.valid_starts_angles[exp_num] for exp_num in exposure_frames_data]
    all_ending_angles = [exposure_frames.valid_ends_angles[exp_num] for exp_num in exposure_frames_data]

    all_starting_angles = np.unique(np.concatenate(all_starting_angles))
    all_ending_angles = np.unique(np.concatenate(all_ending_angles))

    ax = plt.subplot()
    ax.set_xlabel("RA [Degrees]")
    ax.set_ylabel("DEC [Degrees]")
    ax.set_title("Observation Level Plot")
    ax.scatter(ra, dec, marker="X", c="red")
    ax.scatter(
        plotting_catalog["ra"], plotting_catalog["dec"], c="deeppink"
    )

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

    patches = []

    exposure = exposure_frames.data[exposure_id].loc[1]
    region = exposure_frames.get_susceptibility_region(exposure)

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
    """Calculate the targets from a catalog that fall within inner and outer radii."""

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
