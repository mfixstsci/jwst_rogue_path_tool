import matplotlib as mpl
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from pysiaf.utils import rotations

from jwst_rogue_path_tool.utils import get_consecutive_valid_angles, get_intersecting_angles


def create_exposure_plots(program, ncols=3):
    """Generate exposure level plots
    """
    wedge_length = program.catalog_inner_radius - 1

    for obs_id in program.observation_exposure_combos:
        # Dynamically calculate number of rows based on ncols
        nrows = len(program.observation_exposure_combos[obs_id]) // ncols + (len(program.observation_exposure_combos[obs_id]) % ncols > 0)

        for n, exp_num in enumerate(program.observation_exposure_combos[obs_id]):
            exposure = program.exposure_frames[obs_id][exp_num]
            consecutive_angles = exposure.consecutive_angles
            valid_angles = exposure.valid_angles

            # Some exposures will have no valid/consecutive angles.
            if not valid_angles:
                continue

            ax = plt.subplot(nrows, ncols, n + 1)
            ax.set_xlabel('RA [Degrees]')
            ax.set_ylabel('DEC [Degrees]')
            ax.set_title('Observation {}, Exposure: {}'.format(obs_id, exp_num))
            ax.scatter(program.ra, program.dec, marker='X', c='red')
            ax.scatter(program.plotting_catalog['ra'],
                       program.plotting_catalog['dec'], c='deeppink')

            ax.invert_xaxis()

            for angles in consecutive_angles:
                min_theta = min(angles)
                max_theta = max(angles)
                w = Wedge((program.ra, program.dec), wedge_length, min_theta, 
                        max_theta, fill=False, color='darkseagreen', 
                        joinstyle='round')
                ax.add_artist(w)

            for angle in valid_angles:
                exposure.calculate_attitude(angle)
                sus_region_patches = get_susceptibility_region_patch(exposure)

                for patch in sus_region_patches:
                    ax.add_patch(patch)

        plt.tight_layout()
        plt.show()


def create_observation_plot(program, obs_id):
    """Plot that describe all valid angles at the observation level
    """

    obs_level_valid_angles = []
    wedge_length = program.catalog_inner_radius - 1

    for exp_num in program.exposure_frames[obs_id]:
        exposure = program.exposure_frames[obs_id][exp_num]
        exposure.get_valid_angles(program.angle_step)
        valid_angles = exposure.valid_angles
        obs_level_valid_angles.append(valid_angles)

    intersecting_angles = get_intersecting_angles(obs_level_valid_angles)

    obs_level_consecutive_angles = get_consecutive_valid_angles(intersecting_angles, program.angle_step)

    ax = plt.subplot()
    ax.set_xlabel('RA [Degrees]')
    ax.set_ylabel('DEC [Degrees]')
    ax.set_title('Observation Level Plot')
    ax.scatter(program.ra, program.dec, marker='X', c='red')
    ax.scatter(program.plotting_catalog['ra'],
            program.plotting_catalog['dec'], c='deeppink')

    ax.invert_xaxis()

    for angles in obs_level_consecutive_angles:
        min_theta = min(angles)
        max_theta = max(angles)
        w = Wedge((program.ra, program.dec), wedge_length, min_theta, max_theta, 
                fill=False, color='darkseagreen', joinstyle='round')
        ax.add_artist(w)

    plt.tight_layout()
    plt.show()


def get_susceptibility_region_patch(exposure):

    region = exposure.sus_reg

    patches = []

    for key in region:
        module = region[key]
        attitude = exposure.attitude

        v2 = 3600*module.V2V3path.vertices.T[0] # degrees --> arcseconds
        v3 = 3600*module.V2V3path.vertices.T[1] # degrees --> arcseconds

        ra_rads, dec_rads = rotations.tel_to_sky(attitude, v2, v3) # returns ra and dec in radians
        ra_deg, dec_deg = ra_rads*180./np.pi, dec_rads*180./np.pi # convert to degrees

        ra_dec_path = Path(np.array([ra_deg, dec_deg]).T, module.V2V3path.codes)
        ra_dec_patch = PathPatch(ra_dec_path, lw=2, alpha=0.1)
        patches.append(ra_dec_patch)
        
    return patches
