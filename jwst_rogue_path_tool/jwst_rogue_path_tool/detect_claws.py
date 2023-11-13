"""
Module to predict the presence of the stray-light artifacts know as claws
in NIRCam Imaging observations, specified in APT.

Authors
-------
    - Mario Gennaro
"""

import os

import pysiaf
from pysiaf.utils import rotations
from jwst_backgrounds import jbt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from jwst_rogue_path_tool.jwst_rogue_path_tool.apt_sql_parser import AptSqlFile
from jwst_rogue_path_tool.jwst_rogue_path_tool.utils import get_config, querysimbad

SETTINGS = get_config()
DATA_PATH = SETTINGS["jwst_rogue_path_data"]


class AptProgram:

    """
    Class that handles the APT-program-level information.
    It can configure "observation" objects based on the desired observation ids,
    and can cal the observation.check_multiple_angles method to perform
    a check of stars in the susceptibility region for all the exposures of a
    given observation and for multiple observations of a given program
    """

    def __init__(self, sqlfile, instrument="NIRCAM"):
        """
        Parameters
        ----------
        sqlfile : str
            Path to an APT-exported sql file

        instrument : str
            JWST Instrument name
        """

        sql = AptSqlFile(sqlfile)

        self.exposure_table = sql.build_aptsql_database_table("exposures").to_pandas()

        try:
            self.target_table = sql.build_aptsql_database_table(
                "fixed_target"
            ).to_pandas()
        except ValueError:
            raise ValueError("**** This is a moving target program, exiting ****")

        self.nrc_template_table = sql.build_aptsql_database_table(
            "nircam_templates"
        ).to_pandas()
        self.nes_table = sql.build_aptsql_database_table(
            "nircam_exposure_specification"
        ).to_pandas()
        self.visit_table = sql.build_aptsql_database_table("visit").to_pandas()

        self.instrument = instrument
        self.program_id = self.exposure_table.iloc[0]["program"]
        self.siaf = pysiaf.Siaf(self.instrument)
        self.observations = []

    def add_observations(
        self,
        observation_ids=None,
        catalog_args={
            "inner_rad": 8.0,
            "outer_rad": 12.0,
            "sourcecat": "SIMBAD",
            "band": "K",
            "maxmag": 4.0,
            "simbad_timeout": 300,
            "verbose": True,
        },
        small_region=False,
    ):
        """
        Configure multiple observations and append them to the self.observations list

        Parameters
        ----------
        observation_ids: list of Pandas indexes
            IDs of the observations to add. If None, all the opbservations in the prgram
            will be added

        catalog_args: dictionary
            parameters to be passed to the get_catalog method of an observation object
        """

        if observation_ids is None:
            observation_ids = self.exposure_table["observation"].unique()

        # TODO might be able to rewrite loop structure with pandas method or something?
        for observation_id in observation_ids:
            added = False

            for obs in self.observations:
                if observation_id == obs.observation_id:
                    print("observation_id {} already added".format(observation_id))
                    added = True
                    break

            if added == False:
                observation_exists = self.configure_observation(
                    observation_id, catalog_args, small_region=small_region
                )
                if observation_exists is not None:
                    self.observations.append(observation_exists)

    def configure_observation(
        self,
        observation_id,
        catalog_args,
        small_region=False,
    ):
        """
        Create an observation object

        Parameters
        ----------
        observation_id: index of a Pandas dataframe
            Represents the observation ID of interest

        catalog_args: dictionary
            parameters to be passed to the get_catalog method of an observation object

        small_region: boolean
            if True restricts the search to a smaller susceptibility region
            in the rogue path (default: False)

        nrc_exp_spectral_order: integer
            specifies which exposure_spec_order_number to use. If None, the code will
            select the exposure_spec_order_number corresponding to the highest number
            of expected counts (based on photon collecting time and zero point value)
            (default: None)

        Returns
        -------
        An observation object. Returns None if no exposures are present with the input
        observation_id in the APT-exported sql file
        """

        print("Adding observation_id:", observation_id)

        if self.instrument == "NIRCAM":
            instrument_prefix = "NRC"

        skip_observation = False
        nrc_used = False
        nrc_parallel = False

        # Get all visits in the observation.
        boolean_mask = self.visit_table["observation"] == observation_id

        # Get the first visit to obtain the observation information since all visits use the same template.
        first_visit_table = self.visit_table[boolean_mask].iloc[0]

        if (first_visit_table["template"] != "NIRCam Imaging") & (
            first_visit_table["template"] != "NIRCam Wide Field Slitless Spectroscopy"
        ):
            if "template_coord_parallel_1" in self.visit_table.columns:
                if (
                    first_visit_table["template_coord_parallel_1"] != "NIRCam Imaging"
                ) & (
                    first_visit_table["template_coord_parallel_1"]
                    != "NIRCam Wide Field Slitless Spectroscopy"
                ):
                    skip_observation = True

                    if "NIRC" in first_visit_table["template"]:
                        template_name = first_visit_table["template"]
                        nrc_used = True
                    elif "NIRC" in first_visit_table["template_coord_parallel_1"]:
                        template_name = first_visit_table["template_coord_parallel_1"]
                        nrc_used = True

                else:
                    nrc_parallel = True
                    nrc_used = True

            else:
                skip_observation = True

                if "NIRC" in first_visit_table["template"]:
                    template_name = first_visit_table["template"]
                    nrc_used = True

        else:
            nrc_used = True

        if skip_observation:
            if nrc_used:
                print(
                    "**** The {} template is not supported, observation_id {} will not be added ****".format(
                        template_name, observation_id
                    )
                )
            else:
                print(
                    "**** observation_id {} does not use NIRCam and will not be added ****".format(
                        observation_id
                    )
                )
            return None
        else:
            if nrc_parallel:
                print("**** NIRCam imaging used in parallel ****")

        boolean_mask = (
            (self.exposure_table["observation"] == observation_id)
            & (self.exposure_table["pointing_type"] == "SCIENCE")
            & (self.exposure_table["AperName"].str.contains(instrument_prefix))
        )

        if np.sum(boolean_mask) == 0:
            print(
                "No {} expsoures in this program for observation_id {}".format(
                    self.instrument, observation_id
                )
            )
            return None

        print(
            "Total number of {} exposures in this observation: {}".format(
                self.instrument, np.sum(boolean_mask)
            )
        )
        exposure_table_obs = self.exposure_table[boolean_mask]

        boolean_mask = self.nes_table["observation"] == observation_id
        print(
            "Total number of {} exposure specifications: {}".format(
                self.instrument, np.sum(boolean_mask)
            )
        )
        nes_table_obs = self.nes_table[boolean_mask]

        boolean_mask = self.visit_table["observation"] == observation_id
        print("Total number of visits: {}".format(np.sum(boolean_mask)))
        visit_table_obs = self.visit_table[boolean_mask]

        boolean_mask = self.nrc_template_table["observation"] == observation_id
        modules = self.nrc_template_table[boolean_mask].iloc[0]["modules"]

        target_row_idx = (
            self.target_table["target_id"] == exposure_table_obs.iloc[0]["target_id"]
        )
        target_ra = self.target_table.loc[target_row_idx, "ra_computed"].values[0]
        target_dec = self.target_table.loc[target_row_idx, "dec_computed"].values[0]

        return Observation(
            exposure_table_obs,
            nes_table_obs,
            visit_table_obs,
            target_ra,
            target_dec,
            modules,
            catalog_args,
            self.siaf,
            smallregion=small_region,
        )

    def check_observations(
        self, observation_ids=None, angular_step=0.5, rogue_path_padding=0
    ):
        """
        Convenience function to check multiple observations for stars in the
        susceptibility region

        Parameters
        ----------
        observation_id: list of Pandas indexes
            IDs of the observations to check. If None, all the opbservations in the prgram
            will be added

        angular_step: float
            The resolution at which to scan the whole (0,360) range of PAs, in degrees

        rogue_path_padding: float
            Extra padding around the susceptibility region (SR) (stars outside the nominal
            SR, but within rogue_path_padding are flagged as "inside")
        """

        if observation_ids is None:
            observation_ids = self.exposure_table["observation"].unique()

        for observation_id in observation_ids:
            for obs in self.observations:
                if obs.observation_id == observation_id:
                    print("Checking observation_id:", observation_id)
                    obs.check_multiple_angles(angular_step, rogue_path_padding=0.0)


class Observation:

    """
    Class that handles an indivdual observation.
    It contains info on each exposure pointing configuration,
    retrieves a catalog within a certain annulus, and can check for stars
    in the susceptibility region
    """

    def __init__(
        self,
        exposure_table_obs,
        nes_table_obs,
        target_ra,
        target_dec,
        modules,
        catargs,
        siaf,
        smallregion=False,
    ):
        """
        Parameters
        ----------
        exposure_table_obs: Pandas dataframe
            contains one row from the exposure table, per each exposure within the observation

        nes_table_obs: Pandas dataframe
            contains one row from nircam_exposures_specification table for each exposure specification
             selected for this observation

        visit_table_obs: Pandas dataframe
            contains one row from the visit table for each visit associated to this observation


        target_ra,target_dec: floats
            Coorindtaes of the target of the obsrvation in decimal degreess

        modules: list of strings
            name of the nircam modules configuration for the exposures within this observation
            (note that in NIRCam imaging, all the exposures of an observation must have the same configuration)

        catargs: dictionary
            parameters to be passed to the get_catalog method

        siaf: pysiaf.Siaf instance
            object containing the apertures info for NIRCam
        """

        # TODO Make this more pythonic, use args, kwargs?

        self.exposure_table_obs = exposure_table_obs
        self.nes_table_obs = nes_table_obs
        self.observation_id = self.exposure_table_obs.iloc[0]["observation"]
        self.program = self.exposure_table_obs.iloc[0]["program"]
        self.target_ra = target_ra
        self.target_dec = target_dec
        self.modules = modules
        self.catargs = catargs
        self.smallregion = smallregion
        self.SRlist, self.SRnames = self.get_susceptibility_region_list()
        self.efs = self.get_exposure_frames(siaf)
        self.catdf = self.get_catalog()

    def get_susceptibility_region_list(self):
        """
        Parameters
        ----------
        susceptibility_region

        Returns
        -------
        susceptibility_region_list: list
            The list of matplotlib.Path objects for susceptibility regions corresponding to the module
            used in this observation

        susceptibility_region_names: list
            A description of this variable
        """

        # TODO ask Mario about this method, there was a potential bug (unknown variable name)
        # method not called anywhere else inside of this module.

        if (self.modules == "ALL") | (self.modules == "BOTH"):
            susceptibility_region_list = [
                SusceptibilityZoneVertices(module="A", small=self.smallregion),
                SusceptibilityZoneVertices(module="B", small=self.smallregion),
            ]
            susceptibility_region_names = ["A", "B"]
        else:
            if self.modules[0] == "A":
                susceptibility_region_list = [
                    SusceptibilityZoneVertices(module="A", small=self.smallregion)
                ]
                susceptibility_region_names = ["A"]
            elif self.modules[0] == "B":
                susceptibility_region_list = [
                    SusceptibilityZoneVertices(module="B", small=self.smallregion)
                ]
                susceptibility_region_names = ["B"]

        return susceptibility_region_list, susceptibility_region_names

    def get_exposure_frames(self, siaf):
        """
        Parameters
        ----------
        siaf: pysiaf.Siaf
            science instrument aperture file

        Returns
        ----------
        exposure_frames: list of exposure_frame objects
            The list of objects containing pointing info for each exposure within this observation
        """

        exposure_frames = []
        for i, row in self.exposure_table_obs.iterrows():
            boolean_mask = (self.nes_table_obs["visit"] == row["visit"]) & (
                self.nes_table_obs["order_number"] == row["exposure_spec_order_number"]
            )
            exposure_frames.append(
                ExposureFrame(row, siaf, self.nes_table_obs.loc[boolean_mask])
            )

        return exposure_frames

    def get_catalog(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        df: Pandas dataframe
            dataframe containing the coordinates of stars within an annulus centered on the
            observation target. The catalog characterisitics are based on the catargs dictionary

        """

        # Get the maximum relative offset between exposures and pad the catalog search radius
        ra_cen_sorted = np.sort(
            self.exposure_table_obs["ra_center_rotation"].values.astype(np.float_)
        )
        dec_cen_sorted = np.sort(
            self.exposure_table_obs["dec_center_rotation"].values.astype(np.float_)
        )

        max_ra_diff = np.abs(ra_cen_sorted[-1] - ra_cen_sorted[0])
        max_dec_diff = np.abs(dec_cen_sorted[-1] - dec_cen_sorted[0])

        max_delta = np.sqrt(np.sum(np.square([max_ra_diff, max_dec_diff])))

        inner_rad = self.catargs["inner_rad"] - max_delta
        outer_rad = self.catargs["outer_rad"] + max_delta

        if self.catargs["verbose"]:
            print("Adopted inner and outer radius", inner_rad, outer_rad)

        # Retrieve a catalog
        if self.catargs["sourcecat"] == "SIMBAD":
            df_in = querysimbad(
                self.target_ra,
                self.target_dec,
                rad=inner_rad,
                band=self.catargs["band"],
                maxmag=self.catargs["maxmag"],
                simbad_timeout=self.catargs["simbad_timeout"],
            ).to_pandas()
            df_out = querysimbad(
                self.target_ra,
                self.target_dec,
                rad=outer_rad,
                band=self.catargs["band"],
                maxmag=self.catargs["maxmag"],
                simbad_timeout=self.catargs["simbad_timeout"],
            ).to_pandas()

            df = pd.concat([df_in, df_out]).drop_duplicates(keep=False)

            for i, row in df.iterrows():
                coord = SkyCoord(row["RA"], row["DEC"], unit=(u.hourangle, u.deg))
                df.loc[i, "RAdeg"] = coord.ra.deg
                df.loc[i, "DECdeg"] = coord.dec.deg

        if self.catargs["sourcecat"] == "2MASS":
            df = pd.read_csv(self.catargs["2MASS_filename"])
            boolean_mask = df[self.catargs["band"]] < self.catargs["maxmag"]
            df = df[boolean_mask]

            c1 = SkyCoord(self.target_ra * u.deg, self.target_dec * u.deg, frame="icrs")
            c2 = SkyCoord(
                df["ra"].values * u.deg, df["dec"].values * u.deg, frame="icrs"
            )
            sep = c1.separation(c2)
            boolean_mask = (sep.deg < outer_rad) & (sep.deg > inner_rad)

            df = df[boolean_mask]
            df.rename(columns={"ra": "RAdeg", "dec": "DECdeg"}, inplace=True)

        return df

    def check_multiple_angles(
        self, angular_step, filter_short=None, rogue_path_padding=0.0
    ):
        """
        Convenience method to check multiple angles at once, for all
        exposures within this observation

        Parameters
        ----------
        angular_step: float
            The resolution at which to scan the whole (0,360) range of PAs, in degrees

        filter_short: string
            name of the filter in use, will be used to downselect the exposures.
            if None will be defaulted to the first filter in the nircam_expsoures_specification
            table

        rogue_path_padding: float
            Extra padding around the susceptibility region (SR) (stars outside the nominal
            SR, but within rogue_path_padding are flagged as "inside")
        """

        if filter_short is None:
            filter_short = self.nes_table_obs["filter_short"].values[0]

        efs_here = [
            ef
            for ef in self.efs
            if ef.nes_table_row["filter_short"].values[0] == filter_short
        ]

        self.angstep = angular_step
        self.RP_padding = rogue_path_padding
        self.attitudes = np.arange(0.0, 360.0, angular_step)
        self.IN = np.empty(
            [len(self.catdf), self.attitudes.size, len(self.SRlist), len(efs_here)],
            dtype=np.bool_,
        )
        self.V2 = np.empty([len(self.catdf), self.attitudes.size, len(efs_here)])
        self.V3 = np.empty([len(self.catdf), self.attitudes.size, len(efs_here)])
        self.good_angles = np.zeros(
            [self.attitudes.size, len(efs_here)], dtype=np.bool_
        )

        for i, attitude in enumerate(self.attitudes):
            IN_one, V2_one, V3_one, check_one = self.check_one_angle(
                attitude, filter_short, rogue_path_padding=self.RP_padding
            )
            self.V2[:, i, :], self.V3[:, i, :] = V2_one, V3_one
            self.IN[:, i, :, :] = IN_one
            self.good_angles[i, :] = check_one

        V3PA_valid_ranges_starts = []
        V3PA_valid_ranges_ends = []

        for k in range(len(efs_here)):
            change = np.where(self.good_angles[:-1, k] != self.good_angles[1:, k])[0]

            if change.size > 0:
                if self.good_angles[change[0], k]:
                    change = np.roll(change, 1)

                V3PA_valid_ranges_starts.append(self.angstep * change[::2])
                V3PA_valid_ranges_ends.append(self.angstep * change[1::2])
        else:
            V3PA_valid_ranges_starts.append(None)
            V3PA_valid_ranges_ends.append(None)

        self.good_angles_obs = np.all(self.good_angles, axis=1)

        V3PA_valid_ranges_obs_starts = []
        V3PA_valid_ranges_obs_ends = []

        change = np.where(self.good_angles_obs[:-1] != self.good_angles_obs[1:])[0]
        if change.size > 0:
            if self.good_angles_obs[change[0]]:
                change = np.roll(change, 1)

            V3PA_valid_ranges_obs_starts = self.angstep * change[::2]
            V3PA_valid_ranges_obs_ends = self.angstep * change[1::2]
        else:
            V3PA_valid_ranges_obs_starts = None
            V3PA_valid_ranges_obs_ends = None

        self.V3PA_validranges_starts = V3PA_valid_ranges_starts
        self.V3PA_validranges_ends = V3PA_valid_ranges_ends
        self.V3PA_validranges_obs_starts = V3PA_valid_ranges_obs_starts
        self.V3PA_validranges_obs_ends = V3PA_valid_ranges_obs_ends

    def check_one_angle(self, attitude, filter_short, rogue_path_padding=0.0):
        """
        Method to check for the presence of stars in the susceptibility
        region at a fixed angle, for all exposures of a given observation

        Parameters
        ----------
        attitude: float
            position angle to check, in degrees

        filter_short: string
            name of the filter in use, will be used to downselect the exposures

        rogue_path_padding: float
            Extra padding around the susceptibility region (stars outside the nominal
            SR, but within RP_padding are flagged as "inside")

        Returns
        ---------
        IN_one: numpy boolean array (catalog size x number of SR region x number of exposures in the observation)
            True values indicate stars that are in (one of) the SRs for one of the exposures
        V2_one, V3one: numpy array (catalog size x number of exposures in the observation)
            V2, V3 coordinates (jn deg) for a given stars and a given exposure
        check_one: numpy boolean array (number of exposures in the observation)
            True if any star is in either SRs for a given exposure
        """

        efs_here = [
            ef
            for ef in self.efs
            if ef.nes_table_row["filter_short"].values[0] == filter_short
        ]

        IN_one = np.empty(
            [len(self.catdf), len(self.SRlist), len(efs_here)], dtype=np.bool_
        )
        V2_one = np.empty([len(self.catdf), len(efs_here)])
        V3_one = np.empty([len(self.catdf), len(efs_here)])
        check_one = np.zeros(len(efs_here), dtype=np.bool_)

        for k, ef in enumerate(efs_here):
            ef.define_attitude(attitude)
            V2_one[:, k], V3_one[:, k] = ef.V2V3_at_one_attitude(
                self.catdf["RAdeg"], self.catdf["DECdeg"]
            )

            for j, SR in enumerate(self.SRlist):
                IN_one[:, j, k] = SR.V2V3path.contains_points(
                    np.array([V2_one[:, k], V3_one[:, k]]).T, radius=rogue_path_padding
                )
            if len(self.SRlist) > 1:
                if ~(np.any(IN_one[:, 0, k]) | np.any(IN_one[:, 1, k])):
                    check_one[k] = True
            else:
                if ~(np.any(IN_one[:, 0, k])):
                    check_one[k] = True

        return IN_one, V2_one, V3_one, check_one


class ExposureFrame:
    """
    The main class to handle pointing info and rotation for an individual exposure
    """

    def __init__(self, exposure_table_row, siaf, nes_table_row):
        """
        Parameters
        ----------
        exposure_table_row: single row in a Pandas dataframe
            contains the pointing info on this specific exposures

        siaf: instance of pysiaf.Siaf
            used to extract the info on the aperture used in this exposure

        nes_table_row: single row in a Pandas dataframe
            contains info on the expsoure specification from which the exposure
            in question is generated
        """

        self.exposure_table_row = exposure_table_row
        self.nes_table_row = nes_table_row
        self.V2Ref = siaf[self.exposure_table_row["AperName"]].V2Ref
        self.V3Ref = siaf[self.exposure_table_row["AperName"]].V3Ref
        self.raRef = np.float_(self.exposure_table_row["ra_center_rotation"])
        self.decRef = np.float_(self.exposure_table_row["dec_center_rotation"])

    def define_attitude(self, v3pa):
        """
        Define an attitude matrix (pysiaf.rotations)

        Parameters
        ----------
        v3pa:
            position angle of the v3 axis
        """
        self.attitude = rotations.attitude(
            self.V2Ref, self.V3Ref, self.raRef, self.decRef, v3pa
        )

    def V2V3_at_one_attitude(self, radeg, decdeg, verbose=False):
        """
        Compute V2,V3 locations of stars at a given attitude

        Parameters
        ----------
        radeg, decdeg: lists of floats
            stellar coordinates in decimal degrees

        Returns
        ---------
        v2,v3 positions in degrees

        """

        v2rads, v3rads = rotations.sky_to_tel(
            self.attitude, radeg, decdeg, verbose=verbose
        )
        return v2rads.value * 180.0 / np.pi, v3rads.value * 180.0 / np.pi


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
