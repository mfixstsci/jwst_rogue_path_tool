import collections
from copy import deepcopy
import os

from matplotlib.path import Path
import numpy as np
import pandas as pd
import pysiaf
from pysiaf.utils import rotations
from tqdm import tqdm

from jwst_rogue_path_tool.apt_sql_parser import AptSqlFile


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

        self.__sql = AptSqlFile(sqlfile)

        if "fixed_target" not in self.__sql.tablenames:
            raise Exception("JWST Rogue Path Tool only supports fixed targets")

        self.assign_catalog()
        self.get_target_information()
        self.__build_observations()
        self.__build_exposures()

    def __build_observations(self):
        self.observations = Observations(self.__sql)

    def __build_exposures(self):
        """Add exposure classes to APT Program"""
        self.exposure_frames = collections.defaultdict(dict)
        for observation_id in self.observations.observation_number_list:
            if observation_id in self.observations.unusable_observations:
                continue
            else:
                for index, row in self.observations.program_data[observation_id][
                    "exposures"
                ].iterrows():
                    self.exposure_frames[observation_id][index] = ExposureFrame(row)

    def assign_catalog(self, catalog_name="2mass"):
        """Assign Catalog"""
        catalog_names = {"2mass": "two_mass_kmag_lt_5.csv", "simbad": ""}

        if catalog_name not in catalog_names.keys():
            raise Exception("AVAILABLE CATALOG NAMES ARE '2mass' and 'simbad' {} NOT AVAILABLE".format(catalog_name))

        self.catalog_name = catalog_name
        selected_catalog = catalog_names[self.catalog_name]
        project_dirname = os.path.dirname(__file__)
        full_catalog_path = os.path.join(project_dirname, "data", selected_catalog)

        self.catalog = pd.read_csv(full_catalog_path)

    def get_target_information(self):
        """obtain target information based on observation"""

        target_info = self.__sql.build_aptsql_dataframe("fixed_target")

        self.ra = target_info["ra_computed"]
        self.dec = target_info["dec_computed"]

    def sweep_angles(self, observation_id, attitudes):
        ra, dec = self.catalog["ra"], self.catalog["dec"]
        for exposure in self.exposure_frames[observation_id]:
            exposure_table = self.observations.program_data[observation_id]["exposures"]
            template_table = self.observations.program_data[observation_id][
                "nircam_templates"
            ]

            exposure_visit = exposure_table["visit"][exposure]
            exposure_module = template_table.loc[
                template_table["visit"] == exposure_visit
            ]["modules"].values[0]

            if exposure_module == "ALL" or exposure_module == "BOTH":
                sus_reg = {}
                sus_reg["A"] = SusceptibilityRegion(module="A")
                sus_reg["B"] = SusceptibilityRegion(module="B")
            else:
                sus_reg = SusceptibilityRegion(module=exposure_module)

            attitudes_swept = collections.defaultdict(dict)

            print(
                "Sweeping angles {} --> {} for Observation: {} and Exposure: {}".format(
                    min(attitudes), max(attitudes), observation_id, exposure
                )
            )

            for angle in tqdm(attitudes):
                v2, v3 = self.exposure_frames[observation_id][
                    exposure
                ].V2V3_at_one_attitude(ra, dec, angle)

                if isinstance(sus_reg, dict):
                    in_one_a = sus_reg["A"].V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0
                    )

                    in_one_b = sus_reg["B"].V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0
                    )

                    if np.any(in_one_a) | np.any(in_one_b):
                        attitudes_swept[angle]["targets_in"] = True
                        attitudes_swept[angle]["targets_loc"] = [in_one_a, in_one_b]
                    else:
                        attitudes_swept[angle]["targets_in"] = False
                        attitudes_swept[angle]["targets_loc"] = [in_one_a, in_one_b]
                else:
                    in_one = sus_reg.V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0
                    )

                    if np.any(in_one):
                        attitudes_swept[angle]["targets_in"] = True
                        attitudes_swept[angle]["targets_loc"] = in_one
                    else:
                        attitudes_swept[angle]["targets_in"] = False
                        attitudes_swept[angle]["targets_loc"] = in_one

            self.exposure_frames[observation_id][exposure].sweeps = attitudes_swept


class Observations:
    def __init__(self, apt_sql):
        self.__sql = apt_sql
        self.program_data_by_observation()
        self.observation_number_list = self.program_data.keys()
        self.drop_unsupported_observations()

    def drop_unsupported_observations(self):
        """Drop observations and exposures from program data"""
        supported_templates = [
            "NIRCam Imaging",
            "NIRCam Wide Field Slitless Spectroscopy",
        ]
        self.unusable_observations = []

        for observation_id in self.observation_number_list:
            visit_table = self.program_data[observation_id]["visit"]
            templates = visit_table["template"]
            exposure_table = self.program_data[observation_id]["exposures"]

            # If any visits have unsupported templates, this will locate them
            unsupported_templates = visit_table[~templates.isin(supported_templates)]

            # If unsupported templates is empty, NRC is primary
            if unsupported_templates.empty:
                # If template_coord_parallel_1 exists in visit table, check if secondary
                # contains non NRC exposures, remove them
                if "template_coord_parallel_1" in visit_table:
                    aperture_names = exposure_table["AperName"]
                    nrc_visits = aperture_names.str.contains("NRC")
                    self.program_data[observation_id]["exposures"] = exposure_table[
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
                    self.program_data[observation_id]["exposures"] = exposure_table[
                        nrc_visits
                    ]
                else:
                    self.unusable_observations.append(observation_id)
            else:
                self.unusable_observations.append(observation_id)

        # Create seperate data object with unusable observations removed.
        self.supported_observations = deepcopy(self.program_data)
        for observation_id in self.unusable_observations:
            self.supported_observations.pop(observation_id)

    def program_data_by_observation(self):
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

            if table == "exposures":
                df = df.loc[df["apt_label"] != "BASE"]

            for observation_id in df["observation"].unique():
                df_by_program_id = df.loc[df["observation"] == observation_id]
                program_data_by_observation_id[observation_id][table] = df_by_program_id

                program_data_by_observation_id[observation_id][
                    "ra"
                ] = target_information["ra_computed"].values[0]

                program_data_by_observation_id[observation_id][
                    "dec"
                ] = target_information["dec_computed"].values[0]

        self.program_data = program_data_by_observation_id


class ExposureFrame:
    def __init__(self, exposure):
        self.exposure_data = exposure
        self.__exposure_identifiers = [
            "observation",
            "visit",
            "exposure_spec_order_number",
            "dither_point_index",
        ]
        self.calculate_V2_V3_reference()

    def calculate_V2_V3_reference(self, instrument="NIRCAM"):
        """Use pysiaf to obtain V2, V3 reference coordinates

        Parameters
        ----------
        instrumet: str
            JWST Instrument name to generate siaf with
        """

        siaf = pysiaf.Siaf(instrument)
        aperture = self.exposure_data["AperName"]

        self.v2_ref = siaf[aperture].V2Ref
        self.v3_ref = siaf[aperture].V3Ref

    def calculate_attitude(self, v3pa):
        self.attitude = rotations.attitude(
            self.v2_ref,
            self.v3_ref,
            self.exposure_data["ra_center_rotation"],
            self.exposure_data["dec_center_rotation"],
            v3pa,
        )

    def describe_exposure(self):
        print("Unique exposure identifiers:")

        for identifier in self.__exposure_identifiers:
            print("{}: ".format(identifier), self.exposure_data[identifier])

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
