import collections
from copy import deepcopy
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.path import Path
import numpy as np
import pandas as pd
import pysiaf
from pysiaf.utils import rotations
from tqdm import tqdm

from jwst_rogue_path_tool.apt_sql_parser import AptSqlFile
from jwst_rogue_path_tool.utils import get_consecutive_valid_angles


class AptProgram:
    """
    Class that handles the APT-program-level information.
    It can configure "observation" objects based on the desired observation ids,
    and can cal the observation.check_multiple_angles method to perform
    a check of stars in the susceptibility region for all the exposures of a
    given observation and for multiple observations of a given program
    """

    def __init__(self, sqlfile, instrument="NIRCAM", usr_defined_obs=None):
        """
        Parameters
        ----------
        sqlfile : str
            Path to an APT-exported sql file

        instrument : str
            JWST Instrument name

        usr_defined_obs : list like
            List of specific oberservations to load from program
        """

        self.__sql = AptSqlFile(sqlfile)

        if "fixed_target" not in self.__sql.tablenames:
            raise Exception("JWST Rogue Path Tool only supports fixed targets")

        self.usr_defined_obs = usr_defined_obs
        self.observation_exposure_combos = collections.defaultdict(list)

        self.assign_catalog()
        self.get_target_information()
        self.__build_observations()
        self.__build_exposures()

    def __build_observations(self):
        self.observations = Observations(self.__sql, self.usr_defined_obs)

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
                    # Make list of observation/exposure combos for looping later
                    self.observation_exposure_combos[observation_id].append(index)

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

    def locate_targets_in_annulus(self, inner_radius=8.0, outer_radius=12.0):
        """Calculate the targets from a catalog that fall within inner and outer radii.
        """

        # Set coordinates for target and catalog
        target_coordinates = SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='icrs')
        catalog_coordinates = SkyCoord(self.catalog['ra'].values*u.deg, self.catalog['dec'].values*u.deg, frame='icrs')
        
        # Calculate separation from target to all targets in catalog
        separation = target_coordinates.separation(catalog_coordinates)
        mask = (separation.deg < outer_radius) & (separation.deg > inner_radius)

        # Retrieve all targets in masked region above.
        self.plotting_catalog = self.catalog[mask]

    def get_target_information(self):
        """obtain target information based on observation"""

        target_info = self.__sql.build_aptsql_dataframe("fixed_target")

        self.ra = target_info["ra_computed"][0]
        self.dec = target_info["dec_computed"][0]

    def sweep_angles(self, observation_id, attitudes):
        """Sweep supplied attitude angle(s) to determine if exposures contain "bright"
        targets in suseptibility region.

        Parameters
        ----------
        observation_id : int
            APT Observation identifier

        attitudes : list-like
            List of attitude values to sweep through.
        """

        ra, dec = self.catalog["ra"], self.catalog["dec"]

        # Merge tables on visits to obtain module per exposure.
        exposure_table = self.observations.program_data[observation_id]["exposures"]
        template_table = self.observations.program_data[observation_id][
            "nircam_templates"
        ]

        merged = exposure_table.merge(template_table[["visit", "modules"]]).set_index(
            exposure_table.index
        )

        for index, row in merged.iterrows():
            sus_reg ={}
            if row["modules"] == "ALL" or row["modules"] == "BOTH":
                sus_reg["A"] = SusceptibilityRegion(module="A")
                sus_reg["B"] = SusceptibilityRegion(module="B")
            elif row["modules"] == 'A':
                sus_reg["A"] = SusceptibilityRegion(module=row["modules"])
            elif row["modules"] == 'B':
                sus_reg["B"] = SusceptibilityRegion(module=row["modules"])

            attitudes_swept = collections.defaultdict(dict)

            print(
                "Sweeping angles {} --> {} for Observation: {} and Exposure: {}".format(
                    min(attitudes), max(attitudes), observation_id, index
                )
            )

            # Loop through all of the attitude angles to determine if catalog targets
            # are in the the suseptibility region.
            for angle in tqdm(attitudes):
                v2, v3 = self.exposure_frames[observation_id][
                    index
                ].V2V3_at_one_attitude(ra, dec, angle)

                # If sus_reg is dictionary, both instrument modules were used,
                # we need to check both modules for catalog targets.

                # Else only one module was used, only check there.

                # NOTE when both modules are used, `target_in` is a two dimensional
                # list. [module_a, module_b] and is a one dimensional for single modules

                attitudes_swept[angle]["targets_in"] = []
                attitudes_swept[angle]["targets_loc"] = []

                for key in sus_reg.keys():
                    in_one = sus_reg[key].V2V3path.contains_points(
                        np.array([v2, v3]).T, radius=0.0)
                    
                    if np.any(in_one):
                        attitudes_swept[angle]["targets_in"].append(True)
                        attitudes_swept[angle]["targets_loc"].append(in_one)
                    else:
                        attitudes_swept[angle]["targets_in"].append(False)
                        attitudes_swept[angle]["targets_loc"].append(in_one)

            self.exposure_frames[observation_id][index].sweeps = attitudes_swept
            self.exposure_frames[observation_id][index].sus_reg = sus_reg


class Observations:
    def __init__(self, apt_sql, usr_defined_obs=None):
        self.__sql = apt_sql
        self.program_data_by_observation(usr_defined_obs)
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
                        raise Exception(("User defined observation: '{}' not available! "
                                         "Available observations are: {}".format(obs, unique_obs)))
                    else:
                        continue

            if specific_observations:
                observations_list = specific_observations
            else:
                observations_list = unique_obs

            for observation_id in observations_list:
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

    def get_valid_angles(self):
        """Collect valid angles
        """         
        # Build array of all targets_in values.
        targets_in = np.array([self.sweeps.get(angle, {}).get('targets_in') for angle in self.sweeps.keys()])
        # Index location in array contains False
        valid_angles_loc = np.any(~targets_in, axis=1)
        # Transform indices into 
        self.valid_angles = np.where(valid_angles_loc)
        self.valid_angle_data = [self.sweeps[angle] for angle in self.valid_angles[0]]
        self.consecutive_angles = get_consecutive_valid_angles(self.valid_angles)

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
