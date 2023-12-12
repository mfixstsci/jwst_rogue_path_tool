import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from pysiaf.utils import rotations


def compute_line(startx, starty, angle, length):
    anglerad = np.pi / 180.0 * angle
    endx = startx + length * np.cos(anglerad)
    endy = starty + length * np.sin(anglerad)

    return np.array([startx, endx]), np.array([starty, endy])

def plot_observations_checks(
    observation, nrows=2, ncols=3, verbose=True, filtershort=None
):
    """
    Plot some summary results after running observation.check_observations.
    It plots the claws-unaffected angles for each exposure and a summary of
    claws-unaffected angles over the whole observation

    Parameters
    ----------
    observation : class
        Observation class

    nrows : integer
        Number of rows in subplot (default=2)

    ncols : integer
        Number of columns in subplot (default=3)

    filtershort : str
        Name of the short wavelength filter
    """

    if filtershort is None:
        filtershort = observation.nes_table_obs["filter_short"].values[0]

    efs_here = [
        ef
        for ef in observation.efs
        if ef.nes_table_row["filter_short"].values[0] == filtershort
    ]

    # Exposure-level plots
    f1, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))

    for k, (ef, ax) in enumerate(zip(efs_here, axs.reshape(-1))):
        ax.scatter(
            observation.catdf["RAdeg"], observation.catdf["DECdeg"], c="deeppink"
        )
        ax.scatter(observation.target_ra, observation.target_dec, marker="X", c="red")
        ax.scatter(ef.raRef, ef.decRef, marker="X", c="orange")
        ax.axis("equal")
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.invert_xaxis()
        ax.set_title("Expnum: {}".format(k + 1))

        for i, att in enumerate(observation.attitudes):
            if observation.good_angles[i, k] == True:
                ef.define_attitude(att)
                for SR in observation.SRlist:
                    SR_RA, SR_DEC = rotations.tel_to_sky(
                        ef.attitude,
                        3600 * SR.V2V3path.vertices.T[0],
                        3600 * SR.V2V3path.vertices.T[1],
                    )
                    SR_RAdeg, SR_DECdeg = (
                        SR_RA.value * 180.0 / np.pi,
                        SR_DEC.value * 180.0 / np.pi,
                    )
                    RADEC_path = Path(
                        np.array([SR_RAdeg, SR_DECdeg]).T, SR.V2V3path.codes
                    )
                    RADEC_patch = patches.PathPatch(RADEC_path, lw=2, alpha=0.05)
                    ax.add_patch(RADEC_patch)

        draw_angstep = observation.angstep
        for s, e in zip(
            observation.V3PA_validranges_starts[k], observation.V3PA_validranges_ends[k]
        ):
            wd = patches.Wedge(
                (ef.raRef, ef.decRef),
                5.5,
                90 - e - 0.5 * draw_angstep,
                90 - s + 0.5 * draw_angstep,
                width=0.5,
            )
            wd.set(color="darkseagreen")

            ls = compute_line(ef.raRef, ef.decRef, 90 - s + 0.5 * draw_angstep, 5.75)
            le = compute_line(ef.raRef, ef.decRef, 90 - e - 0.5 * draw_angstep, 5.75)
            lm = compute_line(ef.raRef, ef.decRef, 90 - 0.5 * (s + e), 7.0)

            ax.add_artist(wd)
            ax.plot(ls[0], ls[1], color="darkseagreen")
            ax.plot(le[0], le[1], color="darkseagreen")
            ax.text(
                lm[0][1],
                lm[1][1],
                "{}-{}".format(s, e),
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_title("Expnum: {}".format(k + 1))
    f1.suptitle("Obsid: {}".format(observation.observation_id))
    f1.tight_layout()

    # Observation-level plots
    f2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(observation.catdf["RAdeg"], observation.catdf["DECdeg"], c="deeppink")
    ax2.scatter(observation.target_ra, observation.target_dec, marker="X", c="red")
    ax2.axis("equal")
    ax2.set_xlabel("RA")
    ax2.set_ylabel("Dec")
    ax2.invert_xaxis()

    draw_angstep = observation.angstep
    if verbose is True:
        print("*** Valid ranges ****")

    for s, e in zip(
        observation.V3PA_validranges_obs_starts, observation.V3PA_validranges_obs_ends
    ):
        wd = patches.Wedge(
            (observation.target_ra, observation.target_dec),
            5.5,
            90 - e - 0.5 * draw_angstep,
            90 - s + 0.5 * draw_angstep,
            width=0.5,
        )
        wd.set(color="darkseagreen")

        ls = compute_line(
            observation.target_ra,
            observation.target_dec,
            90 - s + 0.5 * draw_angstep,
            5.75,
        )
        le = compute_line(
            observation.target_ra,
            observation.target_dec,
            90 - e - 0.5 * draw_angstep,
            5.75,
        )
        lm = compute_line(
            observation.target_ra, observation.target_dec, 90 - 0.5 * (s + e), 7.0
        )

        ax2.add_artist(wd)
        ax2.plot(ls[0], ls[1], color="darkseagreen")
        ax2.plot(le[0], le[1], color="darkseagreen")
        ax2.text(
            lm[0][1],
            lm[1][1],
            "{}-{}".format(s, e),
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
        )

        if verbose == True:
            print("PA Start -- PA End: {} -- {}".format(s, e))

    ax2.set_title("Summary for obsid {}".format(observation.observation_id))
    f2.tight_layout()

    return f1, f2
