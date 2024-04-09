import collections
from copy import deepcopy
import os

from matplotlib.path import Path
import numpy as np
import pandas as pd
from pysiaf.utils import rotations
from tqdm import tqdm

from jwst_rogue_path_tool.apt_sql_parser import AptSqlFile
from jwst_rogue_path_tool.utils import get_valid_angles_windows


class AptProgram:
    """
    Class that handles the APT-program-level information.
    It can configure "observation" objects based on the desired observation ids,
    and can cal the observation.check_multiple_angles method to perform
    a check of stars in the susceptibility region for all the exposures of a
    given observation and for multiple observations of a given program
    """

    def __init__(self, sqlfile, **kwargs):
        """
        Parameters
        ----------
        sqlfile : str
            Path to an APT-exported sql file

        angle_step : float
            Angle step size when searching for targets in susceptibility region

        instrument : str
            JWST Instrument name

        usr_defined_obs : list like
            List of specific oberservations to load from program
        """

        self.__sql = AptSqlFile(sqlfile)
        if "fixed_target" not in self.__sql.tablenames:
            raise Exception("JWST Rogue Path Tool only supports fixed targets")

        self.angle_step = kwargs.get("angle_step", 1.0)
        self.usr_defined_obs = kwargs.get("usr_defined_obs")
        self.observation_exposure_combos = collections.defaultdict(list)
        self.inner_radius = kwargs.get("inner_radius", 8.0)
        self.outer_radius = kwargs.get("outer_radius", 12.0)

    def __build_observations(self):
        """Convenience method to build observations"""
        self.observations = Observations(self.__sql, self.usr_defined_obs)

    def __build_exposure_frames(self):
        """Add exposure classes to APT Program"""
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                observation = self.observations.data[observation_id]
                exposure_frames = ExposureFrames(observation)
                self.observations.data[observation_id]["exposure_frames"] = (
                    exposure_frames.exposure_frames
                )
                self.observations.data[observation_id]["swept_angles"] = (
                    exposure_frames.swept_angles
                )
                self.observations.data[observation_id]["valid_starts"] = (
                    exposure_frames.valid_starts
                )
                self.observations.data[observation_id]["valid_ends"] = (
                    exposure_frames.valid_ends
                )

    def get_target_information(self):
        """obtain target information based on observation"""

        target_info = self.__sql.build_aptsql_dataframe("fixed_target")

        self.ra = target_info["ra_computed"][0]
        self.dec = target_info["dec_computed"][0]

    def run(self):
        """Run code sequentially"""
        self.get_target_information()
        self.__build_observations()
        self.__build_exposure_frames()

    def write_report(self, filename):
        no_valid_angle_exposures = []
        f = open(filename, "a")
        for obs_id in self.exposure_frames:
            f.write(f"**** Valid Ranges for Observation {obs_id} ****\n")
            all_valid_angles = []
            for exp_num in self.exposure_frames[obs_id]:
                valid_angles = self.exposure_frames[obs_id][exp_num].valid_angles
                if valid_angles:
                    all_valid_angles.append(
                        self.exposure_frames[obs_id][exp_num].consecutive_angles
                    )
                else:
                    no_valid_angle_exposures.append(exp_num)
                    continue
            intersecting_angles = np.unique(all_valid_angles).reshape(-1, 2)
            for min_angle, max_angle in intersecting_angles:
                f.write(
                    f"PA Start -- PA End: {min_angle} -- {max_angle} [step size: {self.angle_step}]\n"
                )
        if no_valid_angle_exposures:
            f.write(f"NO VALID ANGLES FOR EXPOSURES {no_valid_angle_exposures}\n")
        f.close()


class Observations:
    def __init__(self, apt_sql, usr_defined_obs=None):
        self.__sql = apt_sql
        self.program_data_by_observation(usr_defined_obs)
        self.observation_number_list = self.data.keys()
        self.drop_unsupported_observations()

    def drop_unsupported_observations(self):
        """Drop observations and exposures from program data"""
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
                    self.data[observation_id]["exposures"] = exposure_table[
                        nrc_visits
                    ]
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
                    self.data[observation_id]["exposures"] = exposure_table[
                        nrc_visits
                    ]
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
        None

        Returns
        -------
        None
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
    def __init__(self, observation, **kwargs):
        self.assign_catalog()
        self.observation = observation
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

        self.build_exposure_frames()
        self.check_in_susceptibility_region()
        self.get_visibility_windows()

    def assign_catalog(self, catalog_name="2mass"):
        """Assign Catalog"""
        catalog_names = {"2mass": "two_mass_kmag_lt_5.csv", "simbad": ""}

        if catalog_name not in catalog_names.keys():
            raise Exception(
                "AVAILABLE CATALOG NAMES ARE '2mass' and 'simbad' {} NOT AVAILABLE".format(
                    catalog_name
                )
            )

        self.catalog_name = catalog_name
        selected_catalog = catalog_names[self.catalog_name]
        project_dirname = os.path.dirname(__file__)
        full_catalog_path = os.path.join(project_dirname, "data", selected_catalog)

        self.catalog = pd.read_csv(full_catalog_path)

    def build_exposure_frames(self):
        self.orders = self.exposure_frame_table.order_number.unique()

        self.exposure_frames = {}

        for order in self.orders:
            self.exposure_frames[order] = self.exposure_frame_table.loc[
                self.exposure_frame_table.order_number == order
            ]

    def get_susceptibility_region(self, exposure):
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
        self.valid_starts = {}
        self.valid_ends = {}

        for exp_num in self.swept_angles:
            angles_bool = [
                self.swept_angles[exp_num][angle]["targets_in"][0]
                for angle in self.swept_angles[exp_num]
            ]

            change = np.where(angles_bool != np.roll(angles_bool, 1))[0]
            if change.size > 0:
                if angles_bool[change[0]]:
                    change = np.roll(change, 1)
                    starts = angles_bool[change[::2]]
                    ends = angles_bool[change[1::2]]
                else:
                    starts = np.array([])
                    ends = np.array([])
            else:
                starts = np.array([])
                ends = np.array([])
            
            self.valid_starts[exp_num] = starts
            self.valid_ends[exp_num] = ends

                

    def calculate_attitude(self, v3pa):
        self.attitude = rotations.attitude(
            self.exposure_data["v2"],
            self.exposure_data["v3"],
            self.exposure_data["ra_center_rotation"],
            self.exposure_data["dec_center_rotation"],
            v3pa,
        )

    def check_in_susceptibility_region(self):
        ra, dec = self.catalog["ra"], self.catalog["dec"]
        self.swept_angles = {}
        for index, row in self.exposure_frames[1].iterrows():
            self.exposure_data = row
            susceptibility_region = self.get_susceptibility_region(row)
            attitudes_swept = collections.defaultdict(dict)
            attitudes = np.arange(0, 360, 1)

            print(
                "Sweeping angles {} --> {} for Observation: {} and Exposure: {}".format(
                    min(attitudes), max(attitudes), self.observation_number, index
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

                for key in susceptibility_region.keys():
                    in_one = susceptibility_region[key].V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0
                    )

                    if np.any(in_one):
                        attitudes_swept[angle]["targets_in"].append(True)
                        attitudes_swept[angle]["targets_loc"].append(in_one)
                    else:
                        attitudes_swept[angle]["targets_in"].append(False)
                        attitudes_swept[angle]["targets_loc"].append(in_one)

            self.swept_angles[index] = attitudes_swept

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


class SusceptibilityRegion:
    def __init__(self, module, small=False):
        if small:
            self.module_data = {
                "A": np.array(
                    [
                        [
                            2.28483,
                            0.69605,
                            0.43254,
                            0.57463,
                            0.89239,
                            1.02414,
                            1.70874,
                            2.28483,
                            2.28483,
                        ],
                        [
                            10.48440,
                            10.48183,
                            10.25245,
                            10.12101,
                            10.07204,
                            9.95349,
                            10.03854,
                            10.04369,
                            10.48440,
                        ],
                    ]
                ),
                "B": np.array(
                    [
                        [
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
                        ],
                        [
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
                        ],
                    ]
                ),
            }
        else:
            self.module_data = {
                "A": np.array(
                    [
                        [
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
                        ],
                        [
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
                        ],
                    ]
                ),
                "B": np.array(
                    [
                        [
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
                        ],
                        [
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
                        ],
                    ]
                ),
            }

        self.module = module
        self.V2V3path = self.get_path()

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
