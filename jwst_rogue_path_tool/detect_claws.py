"""
This module contains objects that perform the data organizing and analysis
routines for detecting "Claw" anomalies with NIRCam.

The AptProgram class accepts an APT sql file that organizes data with observations,
visits, exposures and parses them into python objects (mainly pandas dataframes).

These data are then used to calculate whether exposures and observations are susceptible
to claw anomalies. We display these data in figures at the exposure and observaiton level.

Authors
-------
    - Mario Gennaro
    - Mees Fix

Use
---
    Routines in this module can be imported as follows:

    >>> from jwst_rogue_path_tool.detect_claws import AptProgram
    >>> filename = "/path/to/sql_apt_file.sql"
    >>> program = AptProgram(filename, angular_step=1, usr_defined_obs=[1])
    >>> program.run()
"""

import collections
from copy import deepcopy
import os

from astropy.io import fits
from matplotlib.path import Path
import numpy as np
import pandas as pd
from pysiaf.utils import rotations
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from jwst_rogue_path_tool.apt_sql_parser import AptSqlFile
from jwst_rogue_path_tool.constants import (
    CATALOG_BANDPASS,
    NIRCAM_ZEROPOINTS,
    SUSCEPTIBILITY_REGION_FULL,
    SUSCEPTIBILITY_REGION_SMALL,
    ZEROPOINT,
)
from jwst_rogue_path_tool.plotting import create_exposure_plots, create_observation_plot


PROJECT_DIRNAME = os.path.dirname(__file__)


class AptProgram:
    """Class that handles the APT-program-level information.
    AptProgram takes the sqlfile input and uses the "Observation" and "ExposureFrame"
    objects to organize data into python objects that can be used for various analyses.
    """

    def __init__(self, sqlfile, angular_step=1.0, usr_defined_obs=None):
        """
        Parameters
        ----------
        sqlfile : str
            Path to an APT-exported sql file

        angular_step : float
            Attitude angle step size used to check if surrounding targets land
            in susceptibility region

        usr_defined_obs : list
            List of specific oberservations to load from program
        """

        self.__sql = AptSqlFile(sqlfile)
        if "fixed_target" not in self.__sql.tablenames:
            raise Exception("JWST Rogue Path Tool only supports fixed targets")

        self.angular_step = angular_step
        self.usr_defined_obs = usr_defined_obs

    def __build_observations(self):
        """Convenience method to build Observation objects"""
        self.observations = Observations(self.__sql, self.usr_defined_obs)

    def __build_exposure_frames(self):
        """Convenience method to build ExposureFrame objects"""
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                observation = self.observations.data[observation_id]
                exposure_frames = ExposureFrames(observation, self.angular_step)
                self.observations.data[observation_id]["exposure_frames"] = (
                    exposure_frames
                )

    def get_target_information(self):
        """Obtain RA and Dec of target from APT SQL file"""
        target_info = self.__sql.build_aptsql_dataframe("fixed_target")

        self.ra = target_info["ra_computed"][0]
        self.dec = target_info["dec_computed"][0]

    def plot_exposures(self, observation_id):
        """Create plot for individual exposures for a given observation. Plot
        will contain targets defined in a specific inner and outer radius
        defined by user. Check `jwst_rogue_path_tool.plotting.create_exposure_plots`
        for more information.

        Parameters
        ----------
        observation_id : int
            Observation id number to generate figures from.
        """
        if observation_id not in self.observations.data.keys():
            raise KeyError(f"{observation_id} IS NOT A VALID OBSERVATION ID")
        else:
            observation = self.observations.data[observation_id]
            create_exposure_plots(observation, self.ra, self.dec)

    def plot_observation(self, observation_id):
        """Create plot at the observation level. The "observation level" is
        defined as all of the valid angles from each exposure combined. Plot
        will contain targets defined in a specific inner and outer radius
        defined by user. Check `jwst_rogue_path_tool.plotting.create_observation_plot`
        for more information.

        Parameters
        ----------
        observation_id : int
            Observation id number to generate figures from.
        """
        if observation_id not in self.observations.data.keys():
            raise KeyError(f"{observation_id} IS NOT A VALID OBSERVATION ID")
        else:
            observation = self.observations.data[observation_id]
            create_observation_plot(observation, self.ra, self.dec)

    def run(self):
        """Convenience method to build AptProgram"""
        self.get_target_information()
        self.__build_observations()
        self.__build_exposure_frames()

    def write_report(self, filename, observation):
        """Write "observation level" report given an observation object.

        Parameters
        ----------
        filename : str
            Name of file to write report into.

        observation : jwst_rogue_path_tool.Observations
            Observation object to extract valid angle data from.
        """
        f = open(filename, "a")
        exposure_frames = observation["exposure_frames"]

        all_starting_angles = [
            exposure_frames.valid_starts_angles[exp_num]
            for exp_num in exposure_frames.data
        ]
        all_ending_angles = [
            exposure_frames.valid_ends_angles[exp_num]
            for exp_num in exposure_frames.data
        ]

        all_starting_angles = np.unique(np.concatenate(all_starting_angles))
        all_ending_angles = np.unique(np.concatenate(all_ending_angles))

        obs_id = observation["visit"]["observation"][0]

        f.write(f"**** Valid Ranges for Observation {obs_id} ****\n")
        for min_angle, max_angle in zip(all_starting_angles, all_ending_angles):
            f.write(f"PA Start -- PA End: {min_angle} -- {max_angle}\n")

        # if no_valid_angle_exposures:
        #     f.write(f"NO VALID ANGLES FOR EXPOSURES {no_valid_angle_exposures}\n")

        f.close()


class Observations:
    """Class the organizes metadata from APT SQL file into python object.
    This object is organized by observation number and contains metadata
    associated with it.
    """

    def __init__(self, apt_sql, usr_defined_obs=None):
        """
        Parameters
        ----------
        apt_sql : jwst_rogue_path_tool.apt_sql_parser.AptSqlFile
            Parsed SQL data into python objects (pandas dataframes)

        usr_defined_obs : list
            List of specific oberservations to load from program
        """
        self.__sql = apt_sql
        self.program_data_by_observation(usr_defined_obs)
        self.observation_number_list = self.data.keys()
        self.drop_unsupported_observations()

    def drop_unsupported_observations(self):
        """Convenience method to drop unsupported observation types. This
        method checks all observations including parallels. All metadata
        is kept and new class attribute `self.supported_observations` is created
        to avoid confusion when processing. `self.supported_observations` are the
        only observation from a program that are analyzed by `jwst_rogue_path_tool`.
        """
        supported_templates = [
            "NIRCam Imaging",
            "NIRCam Wide Field Slitless Spectroscopy",
        ]
        self.unusable_observations = []

        for observation_id in self.observation_number_list:
            visit_table = self.data[observation_id]["visit"]
            templates = visit_table["template"]
            exposure_table = self.data[observation_id]["exposures"]

            # If any visits have unsupported templates, this will locate them
            unsupported_templates = visit_table[~templates.isin(supported_templates)]

            # If unsupported templates is empty, NRC is primary
            if unsupported_templates.empty:
                # If template_coord_parallel_1 exists in visit table, check if secondary
                # contains non NRC exposures, remove them
                if "template_coord_parallel_1" in visit_table:
                    aperture_names = exposure_table["AperName"]
                    nrc_visits = aperture_names.str.contains("NRC")
                    self.data[observation_id]["exposures"] = exposure_table[nrc_visits]
            elif "template_coord_parallel_1" in visit_table:
                # If NRC is not the primary, check to see if NRC is the secondary.
                parallel_templates = visit_table["template_coord_parallel_1"]
                unsupported_parallels = visit_table[
                    ~parallel_templates.isin(supported_templates)
                ]
                if unsupported_parallels.empty:
                    # If NRC is secondary, make sure to remove any exposures
                    # associated with the primary instrument
                    aperture_names = exposure_table["AperName"]
                    nrc_visits = aperture_names.str.contains("NRC")
                    self.data[observation_id]["exposures"] = exposure_table[nrc_visits]
                else:
                    self.unusable_observations.append(observation_id)
            else:
                self.unusable_observations.append(observation_id)

        # Create seperate data object with unusable observations removed.
        self.supported_observations = deepcopy(self.data)
        for observation_id in self.unusable_observations:
            self.supported_observations.pop(observation_id)

    def program_data_by_observation(self, specific_observations=None):
        """Class method to organize APT data by obsevation id

        Parameters
        ----------
        specific_observations : list
            List of observations defined by user.
        """
        program_data_by_observation_id = collections.defaultdict(dict)
        target_information = self.__sql.build_aptsql_dataframe("fixed_target")
        for table in [
            "visit",
            "exposures",
            "nircam_exposure_specification",
            "nircam_templates",
        ]:
            df = self.__sql.build_aptsql_dataframe(table)

            unique_obs = df["observation"].unique()

            if table == "exposures":
                df = df.loc[df["apt_label"] != "BASE"]

            if specific_observations:
                for obs in specific_observations:
                    if obs not in unique_obs:
                        raise Exception(
                            (
                                "User defined observation: '{}' not available! "
                                "Available observations are: {}".format(obs, unique_obs)
                            )
                        )
                    else:
                        continue

            if specific_observations:
                observations_list = specific_observations
            else:
                observations_list = unique_obs

            for observation_id in observations_list:
                df_by_program_id = df.loc[df["observation"] == observation_id]
                program_data_by_observation_id[observation_id][table] = df_by_program_id

                program_data_by_observation_id[observation_id]["ra"] = (
                    target_information["ra_computed"].values[0]
                )

                program_data_by_observation_id[observation_id]["dec"] = (
                    target_information["dec_computed"].values[0]
                )

        self.data = program_data_by_observation_id


class ExposureFrames:
    """Class the organizes data from a single observation (made of exposures)
    into exposure frames. An exposure frame is a group of exposures associated
    with a value in the NRC order specification table. Exposures with the same
    order number are a part of the same exposure frame. This object contains
    """

    def __init__(self, observation, angular_step):
        """
        Parameters
        ----------
        observation : dict
            Dictionary containing data from a single observation

        angular_step : float
            Attitude angle step size used to check if surrounding targets land
            in susceptibility region
        """
        self.assign_catalog()
        self.observation = observation
        self.angular_step = angular_step
        self.observation_number = self.observation["visit"]["observation"][0]
        self.exposure_table = self.observation["exposures"]
        self.template_table = self.observation["nircam_templates"]
        self.nrc_exposure_specification_table = self.observation[
            "nircam_exposure_specification"
        ]

        self.module_by_exposure = self.exposure_table.merge(
            self.template_table[["visit", "modules"]]
        ).set_index(self.exposure_table.index)

        self.exposure_frame_table = pd.merge(
            self.module_by_exposure,
            self.nrc_exposure_specification_table,
            left_on=["exposure_spec_order_number"],
            right_on=["order_number"],
            how="left",
        ).set_index(self.module_by_exposure.index)

        self.build_exposure_frames_data()
        self.check_in_susceptibility_region()
        self.get_visibility_windows()

    def assign_catalog(self, catalog_name="2MASS"):
        """Obtain magnitude selected catalog as pandas dataframe.

        Parameters
        ----------
        catalog_name : str
            Survey name of catalog with star positions and magnitudes [options: 2MASS, SIMBAD]
        """
        catalog_names = {"2MASS": "two_mass_kmag_lt_5.csv", "SIMBAD": ""}

        if catalog_name not in catalog_names.keys():
            raise Exception(
                "AVAILABLE CATALOG NAMES ARE '2MASS' and 'SIMBAD' {} NOT AVAILABLE".format(
                    catalog_name
                )
            )

        self.catalog_name = catalog_name
        selected_catalog = catalog_names[self.catalog_name]
        full_catalog_path = os.path.join(PROJECT_DIRNAME, "data", selected_catalog)

        self.catalog = pd.read_csv(full_catalog_path)

    def build_exposure_frames_data(self):
        """Use exposure table to separate data into exposure frame specific
        pandas dataframes. Resetting the index to combinations of exposure and 
        order number will separate the exposures into exposures associate with
        a specific dither pointing. These tables contain exposures that all
        share the same RA and Dec.
        """
        dither_pointings = self.exposure_frame_table.dither_point_index

        self.data = {}

        exposure_by_order_number = self.exposure_frame_table.set_index(
            ["exposure", "order_number"]
        )

        for idx in dither_pointings:
            self.data[idx] = exposure_by_order_number.loc[idx, :]

    def get_susceptibility_region(self, exposure):
        """Based on the module of an exposure frame, create a SuceptibilityRegion 
        instance.

        Parameters
        ----------
        exposure : pandas.core.series.Series
            A row from an exposure frame table
        """
        sus_reg = {}
        if exposure["modules"] == "ALL" or exposure["modules"] == "BOTH":
            sus_reg["A"] = SusceptibilityRegion(module="A")
            sus_reg["B"] = SusceptibilityRegion(module="B")
        elif exposure["modules"] == "A":
            sus_reg["A"] = SusceptibilityRegion(module=exposure["modules"])
        elif exposure["modules"] == "B":
            sus_reg["B"] = SusceptibilityRegion(module=exposure["modules"])

        return sus_reg

    def get_visibility_windows(self):
        """Method to calculate when a target has entered/exited a 
        susceptibility region.
        """
        self.valid_starts_indices = {}
        self.valid_ends_indices = {}

        self.valid_starts_angles = {}
        self.valid_ends_angles = {}

        for exp_num in self.swept_angles:
            angles_bool = [
                self.swept_angles[exp_num][angle]["targets_in"][0]
                for angle in self.swept_angles[exp_num]
            ]

            change = np.where(angles_bool != np.roll(angles_bool, 1))[0]
            if angles_bool[change[0]]:
                change = np.roll(change, 1)
            starts = change[::2]
            ends = change[1::2]

            self.valid_starts_indices[exp_num] = starts
            self.valid_ends_indices[exp_num] = ends

            self.valid_starts_angles[exp_num] = (starts - 0.5) * self.angular_step
            self.valid_ends_angles[exp_num] = (ends - 0.5) * self.angular_step

    def calculate_attitude(self, v3pa):
        """Calculate attitude matrix given V3 position angle.

        Parameters
        ----------
        v3pa : float
            V3 position angle
        """
        self.attitude = rotations.attitude(
            self.exposure_data["v2"],
            self.exposure_data["v3"],
            self.exposure_data["ra_center_rotation"],
            self.exposure_data["dec_center_rotation"],
            v3pa,
        )

    def check_in_susceptibility_region(self):
        """Method to check if stars from catalog are located in susceptibility
        region per angle of attitude. Angles are 0.0 --> 360.0 degrees in steps
        of `self.angular_step`. This method creates a large dictionary that
        contains contain keys "targets_in" and "targets_loc" for each angle of
        attitude.
        
        For a given angle of attitude, if targets from the catalog fall in the
        susceptibility region, "targets_in" will be True and "targets_loc" are
        the indicies of these stars in the catalog.

        When an exposure frame uses both modules, "targets_in" and "targets_loc"
        are two-dimensional. 

        ```
                         A      B         A      B          A      B
        "targets_in" : [True, True] or [False, True] ... [False, False]
        ```
        """
        ra, dec = self.catalog["ra"], self.catalog["dec"]
        self.swept_angles = {}

        for idx in self.data:
            self.dataframe = self.data[idx]
            self.exposure_data = self.dataframe.loc[1]

            self.susceptibility_region = self.get_susceptibility_region(
                self.exposure_data
            )
            attitudes_swept = collections.defaultdict(dict)
            attitudes = np.arange(0, 360, self.angular_step)

            print(
                "Sweeping angles {} --> {} for Observation: {} and Exposure: {}".format(
                    min(attitudes), max(attitudes), self.observation_number, idx
                )
            )

            # Loop through all of the attitude angles to determine if catalog targets
            # are in the the susceptibility region.
            for angle in tqdm(attitudes):
                v2, v3 = self.V2V3_at_one_attitude(ra, dec, angle)

                # If sus_reg is dictionary, both instrument modules were used,
                # we need to check both modules for catalog targets.

                # Else only one module was used, only check there.

                # NOTE when both modules are used, `target_in` is a two dimensional
                # list. [module_a, module_b] and is a one dimensional for single modules
                attitudes_swept[angle]["targets_in"] = []
                attitudes_swept[angle]["targets_loc"] = []

                for key in self.susceptibility_region.keys():
                    in_one = self.susceptibility_region[key].V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0
                    )

                    if np.any(in_one):
                        attitudes_swept[angle]["targets_in"].append(True)
                        attitudes_swept[angle]["targets_loc"].append(in_one)
                    else:
                        attitudes_swept[angle]["targets_in"].append(False)
                        attitudes_swept[angle]["targets_loc"].append(in_one)

            self.swept_angles[idx] = attitudes_swept

    def V2V3_at_one_attitude(self, ra_degrees, dec_degrees, v3pa, verbose=False):
        """
        Compute V2,V3 locations of stars at a given attitude

        Parameters
        ----------
        ra_degrees, dec_degrees: lists of floats
            stellar coordinates in decimal degrees

        Returns
        ---------
        v2_degrees: float
            V2 position in degrees

        v3_degrees: float
            V2 position in degrees
        """

        self.calculate_attitude(v3pa)
        v2_radians, v3_radians = rotations.sky_to_tel(
            self.attitude, ra_degrees, dec_degrees, verbose=verbose
        )

        v2_degrees = v2_radians.value * 180.0 / np.pi
        v3_degress = v3_radians.value * 180.0 / np.pi

        return v2_degrees, v3_degress


class FixedAngle:
    def __init__(self, observation):
        self.observation = observation
        self.exposure_frame_object = self.observation["exposure_frames"]
        self.catalog_name = self.exposure_frame_object.catalog_name
        self.exposure_frame_table = self.observation.exposure_frame_table

        self.filter_modules_combos = self.exposure_frame_table.groupby(
            ["filter_short", "modules"]
        ).size()

        self.total_exposure_duration = self.get_total_exposure_duration()
        self.get_pupil_from_filter()

    def calculate_absolute_magnitude(self):
        self.catalog = self.observation.exposure_frame_object.catalog
        self.bands = CATALOG_BANDPASS[self.catalog_name]

        fict_mag_A, fict_mag_B = {}, {}
        for band in self.bands:
            fict_mag_A[band] = (
                self.catalog[band] - 2.5 * np.log10(avg_intensity_A) + ZEROPOINT
            )
            fict_mag_B[band] = (
                self.catalog[band] - 2.5 * np.log10(avg_intensity_B) + ZEROPOINT
            )

            fict_mag_A[band].replace(np.inf, 99.0, inplace=True)
            fict_mag_B[band].replace(np.inf, 99.0, inplace=True)

    def get_total_exposure_duration(self):
        total_exposure_duration_table = self.exposure_frame_table.groupby(
            "order_number"
        ).sum()["photon_collecting_duration"]

        return total_exposure_duration_table

    def get_pupil_from_filter(self):
        self.unique_pupils = {}
        for fltr in self.filter_modules_combos["filter_short"]:
            if "+" in fltr:
                splt = filter.split("+")
                pupil = splt[0]
                filter = splt[1]
            elif "_" in fltr:
                splt = filter.split("_")
                pupil = splt[0]
                filter = splt[1]
            else:
                pupil = "CLEAR"

            self.unique_pupils[filter] = pupil

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


class SusceptibilityRegion:
    def __init__(self, module, small=False, smooth=False):
        if small:
            self.module_data = SUSCEPTIBILITY_REGION_SMALL
        else:
            self.module_data = SUSCEPTIBILITY_REGION_FULL

        self.module = module
        self.smooth = smooth
        self.V2V3path = self.get_path()

    def get_intensity(self, V2, V3):
        if self.module == "A":
            filename = "Rogue path NCA.fits"
        else:
            filename = "Rogue path NCB.fits"

        self.filename = "path/to/future/datadir" + filename
        self.fh = fits.getheader(self.filename)

        if self.smooth is not None:
            fd = fits.getdata(self.filename)
            self.fd = gaussian_filter(fd, sigma=self.smooth)
        else:
            self.fd = fits.getdata(self.filename)

        self.fd[:, :60] = 0.0
        self.fd[:, 245:] = 0.0
        self.fd[:85, :] = 0.0
        self.fd[160:, :] = 0.0

        x = (
            (V2 - self.fh["AAXISMIN"])
            / (self.fh["AAXISMAX"] - self.fh["AAXISMIN"])
            * self.fh["NAXIS1"]
        )
        y = (
            (V3 - self.fh["BAXISMIN"])
            / (self.fh["BAXISMAX"] - self.fh["BAXISMIN"])
            * self.fh["NAXIS2"]
        )

        xint = np.floor(x).astype(np.int_)
        yint = np.floor(y).astype(np.int_)

        BM1 = xint < 0
        BM2 = yint < 0
        BM3 = xint >= self.fh["NAXIS1"]
        BM4 = yint >= self.fh["NAXIS2"]
        BM = BM1 | BM2 | BM3 | BM4

        xint[BM] = 0
        yint[BM] = 0

        return self.fd[yint, xint]

    def get_path(self):
        V2list = self.module_data[self.module][0]
        V3list = self.module_data[self.module][1]

        V2list = [-1.0 * v for v in V2list]

        verts = []

        for xx, yy in zip(V2list, V3list):
            verts.append((xx, yy))
        codes = [Path.MOVETO]

        for _ in verts[1:-1]:
            codes.append(Path.LINETO)

        codes.append(Path.CLOSEPOLY)

        return Path(verts, codes)
