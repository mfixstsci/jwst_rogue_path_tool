from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import numpy as np

from jwst_rogue_path_tool.utils import get_consecutive_valid_angles, get_intersecting_angles


def create_exposure_plots(program, ncols=3):
    """Generate exposure level plots
    """

    for obs_id in program.observation_exposure_combos:
         # Dynamically calculate number of rows based on ncols
         nrows = len(program.observation_exposure_combos[obs_id]) // ncols + (len(program.observation_exposure_combos[1]) % ncols > 0)

         for n, exp_num in enumerate(program.observation_exposure_combos[obs_id]):
             program.exposure_frames[obs_id][exp_num].get_valid_angles()
             valid_angles = program.exposure_frames[obs_id][exp_num].valid_angles[0]
             consecutive_angles = get_consecutive_valid_angles(valid_angles)

             ax = plt.subplot(nrows, ncols, n + 1)
             ax.set_xlabel('RA [Degrees]')
             ax.set_ylabel('DEC [Degrees]')
             ax.set_title('Observation {}, Exposure: {}'.format(obs_id, exp_num))
             ax.scatter(program.ra, program.dec, marker='X', c='red')
             ax.scatter(program.plotting_catalog['ra'],
                        program.plotting_catalog['dec'], c='deeppink')

             for angles in consecutive_angles:
                 min_theta = min(angles)
                 max_theta = max(angles)
                 w = Wedge((program.ra, program.dec), 7, min_theta, max_theta, 
                           fill=False, color='darkseagreen')
                 ax.add_artist(w)

         plt.tight_layout()
         plt.show()


def create_observation_plot(program, obs_id):
    """Plot that describe all valid angles at the observation level
    """

    obs_level_valid_angles = []

    for exp_num in program.exposure_frames[obs_id]:
        program.exposure_frames[obs_id][exp_num].get_valid_angles()
        valid_angles = program.exposure_frames[obs_id][exp_num].valid_angles
        obs_level_valid_angles.append(valid_angles)

        intersecting_angles = get_intersecting_angles(obs_level_valid_angles)

    obs_level_consecutive_angles = get_consecutive_valid_angles(intersecting_angles)

    ax = plt.subplot()
    ax.set_xlabel('RA [Degrees]')
    ax.set_ylabel('DEC [Degrees]')
    ax.set_title('Observation Level Plot')
    ax.scatter(program.ra, program.dec, marker='X', c='red')
    ax.scatter(program.plotting_catalog['ra'],
            program.plotting_catalog['dec'], c='deeppink')

    for angles in obs_level_consecutive_angles:
        min_theta = min(angles)
        max_theta = max(angles)
        w = Wedge((program.ra, program.dec), 7, min_theta, max_theta, 
                fill=False, color='darkseagreen')
        ax.add_artist(w)

    plt.tight_layout()
    plt.show()
