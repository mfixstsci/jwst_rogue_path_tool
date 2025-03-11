"""
This module contains objects that perform the data organizing and analysis
routines for detecting "Claw" anomalies with NIRCam.

The AptProgram class accepts an APT JSON file that organizes data with observations,
visits, exposures and parses them into python objects (mainly pandas dataframes).

These data are then used to calculate whether exposures and observations are susceptible
to claw anomalies. We display these data in figures at the exposure and observaiton level.

Authors
-------
    - Mario Gennaro
    - Mees Fix
"""

import collections
from copy import deepcopy
from itertools import chain, groupby
import operator
import os

from astropy.io import fits
from matplotlib.path import Path
import numpy as np
import pandas as pd
import pathlib
from pysiaf.utils import rotations
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from jwst_rogue_path_tool.program_data_parser import aptJsonFile
from jwst_rogue_path_tool.constants import (
    PROJECT_DIRNAME,
    SUSCEPTIBILITY_REGION_FULL,
    SUSCEPTIBILITY_REGION_SMALL,
)
from jwst_rogue_path_tool.fixed_angle import fixedAngle
from jwst_rogue_path_tool.plotting import (
    create_exposure_plots,
    create_observation_plot,
    create_v3pa_vs_flux_plot,
)
from jwst_rogue_path_tool.utils import (
    calculate_background,
    get_pupil_from_filter,
    get_pivot_wavelength,
    make_output_directory,
)


class aptProgram:
    """Class that handles the APT-program-level information.
    aptProgram takes the JSON file input and uses the "observations" and "exposureFrame"
    objects to organize data into python objects that can be used for various analyses.
    """

    def __init__(
        self,
        jsonfile,
        angular_step=1.0,
        usr_defined_obs=None,
        usr_defined_angles=None,
        bkg_params=[
            {"threshold": 0.1, "function": np.min},
            {"threshold": 0.2, "function": np.mean},
        ],
    ):
        """
        Parameters
        ----------
        jsonfile : str
            Path to an APT-exported JSON file

        angular_step : float
            Attitude angle step size used to check if surrounding targets land
            in susceptibility region (default: 1.0)

        usr_defined_obs : list
            List of specific oberservations to load from program

        usr_defined_angles : list
            List of angles to valid, only the angle values in this list will be
            evaluated

        bkg_params : dictionary
            Dictionary of background threshold values and function to apply to background
            calculations. "{threshold": 0.1, "function: np.min}" will calculate the value of 10%
            of the minimum background based on JWST pointing. Docs for jwst_backgrounds
            https://github.com/spacetelescope/jwst_backgrounds
        """

        self.json = aptJsonFile(jsonfile)
        if "fixed_target" not in self.json.tablenames:
            raise Exception("JWST Rogue Path Tool only supports fixed targets")

        self.angular_step = angular_step
        self.usr_defined_obs = usr_defined_obs

        # Users may only want to check a list of specific angles of attitude,
        # if they provide a list of angles, those are the only angles that are checked.
        if angular_step and usr_defined_angles:
            raise Exception(
                "You defined an angular step that will check all valid angles (0 --> 359.0) \
                in steps of {angular_step} and usr_defined_angles of {usr_defined_angles} which \
                only checks for targets at these angles of attitude. Please select which you \
                prefer, but both options can not be true."
            )
        elif self.angular_step:
            self.angles_list = np.arange(0, 360.0, self.angular_step)
        elif usr_defined_angles:
            self.angles_list = usr_defined_angles
        else:
            raise Exception(
                "angular_step and usr_defined_angs are both None. You must pass angular_step \
                to check for targets in attitude angles (0.0 --> 359.0) in steps of `angular_step` \
                or a list of individual angles to check `usr_defined_angles` i.e. [20.0, 39.0, 256.0]"
            )
        self.bkg_parameters = bkg_params

    def __build_observations(self):
        """Convenience method to build Observation objects"""
        self.observations = observations(self.json, self.usr_defined_obs)

    def __build_exposure_frames(self):
        """Convenience method to build ExposureFrame objects"""
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                observation = self.observations.data[observation_id]
                exposure_frames = exposureFrames(
                    observation, self.angles_list, self.angular_step
                )
                self.observations.data[observation_id]["exposure_frames"] = (
                    exposure_frames
                )

    def __calculate_averages(self):
        """Convenience method to average out the intensities and positions
        for an observation.
        """
        print("... Averaging intensities per angle ...")
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                averages = collections.defaultdict(dict)
                observation = self.observations.data[observation_id]
                swept_angles = observation["exposure_frames"].swept_angles
                exposure_frame_ids = observation["exposure_frames"].data.keys()
                modules = observation["exposure_frames"].susceptibility_region.keys()
                for angle in self.angles_list:
                    for module in modules:
                        for quantity in ["intensity", "v2", "v3"]:
                            averages[angle][f"avg_{quantity}_{module}"] = np.mean(
                                [
                                    swept_angles[exp_frm][angle][f"{quantity}_{module}"]
                                    for exp_frm in exposure_frame_ids
                                ],
                                axis=0,
                            )
                        observation[f"averages_{module}"] = averages

    def __get_flux_vs_angle(self):
        """Get all flux values as function of angle"""
        print("... Calculating flux values ...")
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                final_fluxes = collections.defaultdict(dict)
                observation = self.observations.data[observation_id]

            total_exposure_durations = observation[
                "exposure_frames"
            ].total_exposure_duration_table

            observation["filters"] = total_exposure_durations["filters"]
            observation["pupils"] = total_exposure_durations["pupils"]

            counts_per_filter = [
                fixedAngle(observation, angle).total_counts
                for angle in self.angles_list
            ]
            flux_keys = list(
                set(chain.from_iterable(sub.keys() for sub in counts_per_filter))
            )

            for key in flux_keys:
                flux_per_key = list(map(operator.itemgetter(key), counts_per_filter))
                final_fluxes["total_counts"][key] = np.array(flux_per_key)

                for filter in observation["filters"]:
                    exposure_duration = total_exposure_durations.loc[
                        (total_exposure_durations["filters"] == filter)
                    ]["photon_collecting_duration"].values[0]
                    if filter in key:
                        pix_dn_ks = (
                            final_fluxes["total_counts"][key] * 1000 / exposure_duration
                        )
                        flux_key = key.replace("total_counts", "dn_pix_ks")
                        final_fluxes["dn_pix_ks"][flux_key] = pix_dn_ks

            observation["flux"] = final_fluxes

    def __find_background_thresholds(self):
        """Find fluxes that are higher than threshold values."""
        print("... Calculating backgrounds and thresholds ...")
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                observation = self.observations.data[observation_id]
                susceptibility_regions = observation[
                    "exposure_frames"
                ].susceptibility_region
                modules = susceptibility_regions.keys()
                fluxes = observation["flux"]["dn_pix_ks"]  # Get flux dictionary
                total_exposure_duration_table = observation[
                    "exposure_frames"
                ].total_exposure_duration_table
                flux_boolean = collections.defaultdict(dict)
                for parameters in self.bkg_parameters:
                    above_threshold = []
                    threshold = parameters["threshold"]
                    statistic_function = parameters["function"]
                    # Calculate the background for each filter/pupil combination
                    for module in modules:
                        for index, row in total_exposure_duration_table.iterrows():
                            pupil = row["pupils"]
                            filter = row["filters"]
                            pivot_wavelength = get_pivot_wavelength(pupil, filter)
                            background = calculate_background(
                                self.ra, self.dec, pivot_wavelength, threshold
                            )

                            wavelengths = background.bathtub["total_thiswave"]
                            lam_thresh = (
                                threshold
                                * statistic_function(wavelengths)
                                / pivot_wavelength
                                * 1000.0
                            )

                            flux_key = f"dn_pix_ks_{pupil}+{filter}_{module}"

                            flux_above_limit = np.copy(fluxes[flux_key])
                            limit_indices = flux_above_limit < lam_thresh

                            flux_boolean_key = f"{filter}_{module}"
                            statistics_key = f"{statistic_function.__name__}_{lam_thresh}_{threshold}"
                            flux_boolean[flux_boolean_key][statistics_key] = (
                                limit_indices
                            )

                            above_threshold.append(limit_indices)

                        # Convert list to an array and then see where all indices are true
                        # for each filter/pupil combination.
                        all_boolean = np.array(above_threshold).all(0)
                        module_boolean_key = (
                            f"flux_boolean_{statistic_function.__name__}_{module}"
                        )
                        flux_boolean[module_boolean_key] = all_boolean

            observation["flux_boolean"] = flux_boolean

    def get_target_information(self):
        """Obtain RA and Dec of target from APT JSON file"""
        target_info = self.json.build_dataframe("fixed_target")

        self.ra = target_info["ra_computed"][0]
        self.dec = target_info["dec_computed"][0]

    def plot_exposures(self, observation, output_directory=None):
        """Create plot for individual exposures for a given observation. Plot
        will contain targets defined in a specific inner and outer radius
        defined by user. Check `jwst_rogue_path_tool.plotting.create_exposure_plots`
        for more information.

        Parameters
        ----------
        observation_id : int
            Observation id number to generate figures from.
        """
        create_exposure_plots(observation, self.ra, self.dec, output_directory)

    def plot_observation(self, observation, output_directory=None):
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
        create_observation_plot(observation, self.ra, self.dec, output_directory)

    def plot_v3pa_vs_flux(self, observation, output_directory=None):
        """Create plot of position angle vs flux. This plot will only work if users
        use the angular_step keyword calculate the flux for all 360 degrees of attitude.
        """

        if self.angular_step:
            create_v3pa_vs_flux_plot(observation, output_directory)
        else:
            raise Exception(
                "PLOTTING V3PA VS FLUX ONLY WORKS FOR OBSERVATIONS WITH ALL 360 DEGREES \
                OF ATTITUDE. SET ANGULAR STEP AND RE-RUN JWST ROGUE PATH TOOL."
            )

    def run(self):
        """Convenience method to build AptProgram"""
        self.get_target_information()
        self.__build_observations()
        self.__build_exposure_frames()
        self.__calculate_averages()
        self.__get_flux_vs_angle()
        self.__find_background_thresholds()

    from itertools import groupby

    def make_background_report(self, observation, output_directory):
        """Write reports for background restricted position angles (PA's).
        This will create a report for the module(s) in the observation provided, this
        means that the resulting PA's will be PA's that are flagged across
        all filters in the observaion for that module.

        observation : dict
            Observation out of observations.data attribute

        output_directory : str
            Output directory path
        """
        obs_id = observation["visit"]["observation"].values[0]
        program = observation["nircam_templates"]["program"].values[0]
        modules = observation["exposure_frames"].susceptibility_region.keys()
        functions = [params["function"].__name__ for params in self.bkg_parameters]
        thresholds = [params["threshold"] for params in self.bkg_parameters]

        path = pathlib.Path(output_directory)
        path = path / str(program) / "valid_angle_reports"
        make_output_directory(path)

        if path.is_dir():
            if len(modules) > 1:
                data = {}
                filename = (
                    f"program_{program}_observation_{obs_id}_background_module_A_B.txt"
                )
                full_file_path = path / filename
                f = open(full_file_path, "w")
                for threshold, function in zip(thresholds, functions):
                    data["A"] = observation["flux_boolean"][
                        f"flux_boolean_{function}_{'A'}"
                    ].nonzero()[0]

                    data["B"] = observation["flux_boolean"][
                        f"flux_boolean_{function}_{'B'}"
                    ].nonzero()[0]

                    valid_pa_angles = np.intersect1d(data["A"], data["B"])

                    f.write("**** Ranges Not Impacted by Background Thresholds ****\n")
                    f.write("**** Modules A + B ****\n")
                    f.write(f"**** Ranges Under {threshold} of {function}  ****\n")
                    for _, g in groupby(
                        enumerate(valid_pa_angles), lambda k: k[0] - k[1]
                    ):
                        start = next(g)[1]
                        end = list(v for _, v in g) or [start]
                        f.write(f"PA Start -- PA End: {start} -- {end[-1]}\n")
            else:
                single_module = list(modules)[0]
                filename = f"program_{program}_observation_{obs_id}_background_module_{single_module}.txt"
                full_file_path = path / filename
                f = open(full_file_path, "w")
                for threshold, function in zip(thresholds, functions):
                    valid_pa_angles = observation["flux_boolean"][
                        f"flux_boolean_{function}_{single_module}"
                    ].nonzero()[0]

                    f.write("**** Ranges Not Impacted by Background Thresholds ****\n")
                    f.write(f"**** Module {single_module} ****\n")
                    f.write(f"**** Ranges Under {threshold} of {function}  ****\n")
                    for _, g in groupby(
                        enumerate(valid_pa_angles), lambda k: k[0] - k[1]
                    ):
                        start = next(g)[1]
                        end = list(v for _, v in g) or [start]

                        f.write(f"PA Start -- PA End: {start} -- {end[-1]}\n")
            print(f"WROTE FILE TO: {full_file_path}")
            f.close()
        else:
            raise Exception(f"CAN NOT WRITE TO {path}, NOT A VALID DIRECTORY")

    def make_report(self, observation, output_directory):
        """Display of write "observation level" report given an observation in a program.

        Parameters
        ----------
        observation : dict
            Observation out of observations.data attribute

        output_directory : str
            Output directory path
        """

        obs_id = observation["visit"]["observation"].values[0]
        program = observation["nircam_templates"]["program"].values[0]

        all_starting_angles = observation["valid_starts_angles"]
        all_ending_angles = observation["valid_ends_angles"]

        path = pathlib.Path(output_directory)
        path = path / str(program) / "valid_angle_reports"
        make_output_directory(path)

        if path.is_dir():
            filename = f"program_{program}_observation_{obs_id}_valid_angle_report.txt"

            full_file_path = path / filename

            f = open(full_file_path, "w")
            f.write(f"**** Valid Ranges for Observation {obs_id} ****\n")
            for min_angle, max_angle in zip(all_starting_angles, all_ending_angles):
                f.write(f"PA Start -- PA End: {min_angle} -- {max_angle}\n")
            f.close()
            print(f"WROTE REPORT TO {full_file_path}")

        else:
            raise Exception(f"CAN NOT WRITE TO {path}, NOT A VALID DIRECTORY")


class observations:
    """Class the organizes metadata from APT JSON file into python object.
    This object is organized by observation number and contains metadata
    associated with it.
    """

    def __init__(self, apt_json, usr_defined_obs=None):
        """
        Parameters
        ----------
        apt_json : jwst_rogue_path_tool.program_data_parser.aptJsonFile
            Parsed JSON data into python objects (pandas dataframes)

        usr_defined_obs : list
            List of specific oberservations to load from program
        """
        self.json = apt_json
        self.program_data_by_observation(usr_defined_obs)
        self.observation_number_list = self.data.keys()
        self.drop_unsupported_observations()

    def drop_unsupported_observations(self):
        """Convenience method to drop unsupported observation types. This
        method checks all observations including parallels. All metadata
        is kept and new class attribute `self.supported_observations` which is created
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
        target_information = self.json.build_dataframe("fixed_target")
        for table in [
            "visit",
            "exposures",
            "nircam_exposure_specification",
            "nircam_templates",
        ]:
            df = self.json.build_dataframe(table)

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


class exposureFrames:
    """Class the organizes data from a single observation (made of exposures)
    into exposure frames. An exposure frame is a group of exposures associated
    with a value in the NRC order specification table. Exposures with the same
    order number are a part of the same exposure frame. This object contains
    """

    def __init__(self, observation, attitudes, angular_step=None):
        """
        Parameters
        ----------
        observation : dict
            Dictionary containing data from a single observation

        attitudes : list like
            Angles of attitude to check for targets falling in the susceptibility
            region.
        """
        self.assign_catalog()
        self.observation = observation
        self.attitude_angles = attitudes
        self.angular_step = angular_step
        self.observation_number = self.observation["visit"]["observation"].values[0]
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
        )

        self.get_total_exposure_duration()
        self.build_exposure_frames_data()
        self.check_in_susceptibility_region()

        # For sweeping all angles we will need a angular_step defined.
        # Else, we have a defined set of angles that are user defined and
        # do not need windows of visibility calculated.
        if self.angular_step:
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
        dither_pointings = self.exposure_frame_table.dither_point_index.unique()
        exposures = self.module_by_exposure.index

        self.data = {}

        for dp, exp in zip(dither_pointings, exposures):
            self.data[exp] = self.exposure_frame_table[
                self.exposure_frame_table["dither_point_index"] == dp
            ]

    def get_total_exposure_duration(self):
        total_exposure_duration_table = (
            self.exposure_frame_table.groupby("filter_short")
            .sum()["photon_collecting_duration"]
            .reset_index()
        )

        filters = total_exposure_duration_table["filter_short"]
        pupils = get_pupil_from_filter(filters)

        total_exposure_duration_table["filters"] = pupils.keys()
        total_exposure_duration_table["pupils"] = pupils.values()

        self.total_exposure_duration_table = total_exposure_duration_table

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
            sus_reg["A"] = susceptibilityRegion(module="A", smooth=5)
            sus_reg["B"] = susceptibilityRegion(module="B", smooth=5)
        elif exposure["modules"] == "A":
            sus_reg["A"] = susceptibilityRegion(module=exposure["modules"], smooth=5)
        elif exposure["modules"] == "B":
            sus_reg["B"] = susceptibilityRegion(module=exposure["modules"], smooth=5)

        return sus_reg

    def get_visibility_windows(self):
        """Method to calculate when a target has entered/exited a
        susceptibility region. This is done at the exposure and
        observation levels.
        """

        # Begin calculating exposure level angles
        self.valid_starts_indices = {}
        self.valid_ends_indices = {}

        self.valid_starts_angles = {}
        self.valid_ends_angles = {}

        angles_bool_obs = []
        for exp_num in self.swept_angles:
            angles_bool = [
                self.swept_angles[exp_num][angle]["targets_in"][0]
                for angle in self.swept_angles[exp_num]
            ]

            angles_bool_obs.append(angles_bool)

            change = np.where(angles_bool != np.roll(angles_bool, 1))[0]
            # Include cases where target is in susceptibility regions
            # for every attitude angle.
            if change.size == 0:
                self.valid_starts_angles[exp_num] = None
                self.valid_ends_angles[exp_num] = None
                continue
            else:
                if angles_bool[change[0]]:
                    change = np.roll(change, 1)

            starts = change[::2]
            ends = change[1::2]

            self.valid_starts_indices[exp_num] = starts
            self.valid_ends_indices[exp_num] = ends

            self.valid_starts_angles[exp_num] = (starts - 0.5) * self.angular_step
            self.valid_ends_angles[exp_num] = (ends - 0.5) * self.angular_step

        # Begin calculating observation level angles
        angles_bool_obs = np.array(angles_bool_obs)
        angles_bool_obs = np.all(angles_bool_obs, axis=0)
        change_obs = np.where(angles_bool_obs != np.roll(angles_bool_obs, 1))[0]

        if change_obs.size == 0:
            self.observation["valid_starts_angles"] = None
            self.observation["valid_ends_angles"] = None
        else:
            if angles_bool_obs[change[0]]:
                change_obs = np.roll(change_obs, 1)

        starts = change_obs[::2]
        ends = change_obs[1::2]

        self.observation["valid_starts_angles"] = (starts - 0.5) * self.angular_step
        self.observation["valid_ends_angles"] = (ends - 0.5) * self.angular_step

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

                         A      B         A      B          A      B
        "targets_in" : [True, True] or [False, True] ... [False, False]
        """
        ra, dec = self.catalog["ra"], self.catalog["dec"]
        self.swept_angles = {}

        for obs_num, obs_data in self.data.items():
            self.dataframe = obs_data
            self.exposure_data = self.dataframe.iloc[0]
            self.susceptibility_region = self.get_susceptibility_region(
                self.exposure_data
            )
            attitudes_swept = collections.defaultdict(dict)

            print(
                "Sweeping angles {} --> {} for Observation: {} and Exposure: {}".format(
                    min(self.attitude_angles),
                    max(self.attitude_angles),
                    self.observation_number,
                    obs_num,
                )
            )

            # Loop through all of the attitude angles to determine if catalog targets
            # are in the the susceptibility region.
            for angle in tqdm(self.attitude_angles):
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

                    # Since we are already getting V2 & V3, use this opportunity to get
                    # claw intensities based on attitude angle.
                    claw_intensity = self.susceptibility_region[key].get_intensity(
                        v2, v3
                    )
                    attitudes_swept[angle][f"intensity_{key}"] = claw_intensity
                    attitudes_swept[angle][f"v2_{key}"] = v2
                    attitudes_swept[angle][f"v3_{key}"] = v3
            # Store swept angles and values.
            self.swept_angles[obs_num] = attitudes_swept

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


class susceptibilityRegion:
    """Class that describes the JWST NRC susceptibility regions. Creates region and
    calculates intensities based on magnitude and location of target that falls in
    susceptibility region.

    Parameters
    ----------
    module : str
        Name of JWST NRC module ("A or "B")

    small : bool
        Create smaller susceptibility region (default: False)

    smooth : bool
        Smooth data with Gaussian Filter (default: False)
    """

    def __init__(self, module, small=False, smooth=False):
        if small:
            self.module_data = SUSCEPTIBILITY_REGION_SMALL
        else:
            self.module_data = SUSCEPTIBILITY_REGION_FULL

        self.module = module
        self.smooth = smooth
        self.get_intensity_map()
        self.V2V3path = self.get_path()
        self.calculate_centroid()

    def calculate_centroid(self):
        """Calculate the centroid of a susceptibility region polygons."""
        vertices = np.array(self.verts)
        num_of_vertices = len(vertices)
        self.centroid = (
            sum(vertices[:, 0]) / num_of_vertices,
            sum(vertices[:, 1]) / num_of_vertices,
        )

    def get_intensity_map(self):
        """Open intensity map reference file"""
        if self.module == "A":
            filename = "rogue_path_nrca.fits"
        elif self.module == "B":
            filename = "rogue_path_nrcb.fits"
        else:
            ValueError(
                f"{self.module} IS NOT A VALID MODULE, VALID MODULES ARE 'A' or 'B'"
            )

        full_file_path = os.path.join(PROJECT_DIRNAME, "data", filename)
        self.fits_header = fits.getheader(full_file_path)

        if self.smooth is not None:
            fits_data = fits.getdata(full_file_path)
            self.fits_data = gaussian_filter(fits_data, sigma=self.smooth)
        else:
            self.fits_data = fits.getdata(full_file_path)

    def get_intensity(self, V2, V3):
        """Calculate the intensity of claw caused by star falling in susceptibility region."""
        self.fits_data[:, :60] = 0.0
        self.fits_data[:, 245:] = 0.0
        self.fits_data[:85, :] = 0.0
        self.fits_data[160:, :] = 0.0

        x = (
            (V2 - self.fits_header["AAXISMIN"])
            / (self.fits_header["AAXISMAX"] - self.fits_header["AAXISMIN"])
            * self.fits_header["NAXIS1"]
        )
        y = (
            (V3 - self.fits_header["BAXISMIN"])
            / (self.fits_header["BAXISMAX"] - self.fits_header["BAXISMIN"])
            * self.fits_header["NAXIS2"]
        )

        xint = np.floor(x).astype(np.int_)
        yint = np.floor(y).astype(np.int_)

        BM1 = xint < 0
        BM2 = yint < 0
        BM3 = xint >= self.fits_header["NAXIS1"]
        BM4 = yint >= self.fits_header["NAXIS2"]
        BM = BM1 | BM2 | BM3 | BM4

        xint[BM] = 0
        yint[BM] = 0

        return self.fits_data[yint, xint]

    def get_path(self):
        """Calculate rogue path for plotting."""
        V2list = self.module_data[self.module][0]
        V3list = self.module_data[self.module][1]

        V2list = [-1.0 * v for v in V2list]

        self.verts = []

        for xx, yy in zip(V2list, V3list):
            self.verts.append((xx, yy))
        self.codes = [Path.MOVETO]

        for _ in self.verts[1:-1]:
            self.codes.append(Path.LINETO)

        self.codes.append(Path.CLOSEPOLY)

        return Path(self.verts, self.codes)
