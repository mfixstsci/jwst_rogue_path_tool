import collections
import numpy as np
import pandas as pd
import pysiaf
from pysiaf.utils import rotations

from jwst_rogue_path_tool.jwst_rogue_path_tool.apt_sql_parser import AptSqlFile


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

        self.program_data_by_observation()
        self.observation_number_list = self.program_data.keys()
        self.drop_unsupported_observations()
        self.construct_exposure_dataframes()

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
        for table in ["visit", "exposures", "nircam_exposure_specification"]:
            df = self.__sql.build_aptsql_dataframe(table)
            for observation_id in df["observation"].unique():
                df_by_program_id = df.loc[df["observation"] == observation_id]
                program_data_by_observation_id[observation_id][table] = df_by_program_id

        self.program_data = program_data_by_observation_id

    def drop_unsupported_observations(self):
        supported_templates = [
            "NIRCam Imaging",
            "NIRCam Wide Field Slitless Spectroscopy",
        ]
        self.unusable_observations = []

        for observation_id in self.observation_number_list:
            visit_table = self.program_data[observation_id]["visit"]
            templates = visit_table["template"]

            # If any visits have unsupported templates, this will locate them
            unsupported_templates = visit_table[~templates.isin(supported_templates)]

            # if empty, all templates were supported!
            # elif a parallel template column exists, locate rows with unsupported parallels
            # ... if no unsupported parallels, then NIRCAM is used, else, no supported modes, drop
            # else there are unsupported templates present and no parallels, drop observation.
            if unsupported_templates.empty:
                print(
                    "Observation {} contains supported templates".format(observation_id)
                )
                print(
                    "Observation {} visit templates:\n{}".format(
                        observation_id, templates
                    )
                )
            elif "template_coord_parallel_1" in visit_table:
                parallel_templates = visit_table["template_coord_parallel_1"]
                unsupported_parallels = visit_table[
                    ~parallel_templates.isin(supported_templates)
                ]
                if unsupported_parallels.empty:
                    print("Observation {} contains supported parallel templates")
                    print(
                        "Observation {} visit parallel templates:\n{}".format(
                            observation_id, parallel_templates
                        )
                    )
                else:
                    print(
                        "No supported primary or parallel templates for Observation {}".format(
                            observation_id
                        )
                    )
                    self.unusable_observations.append(observation_id)
            else:
                print(
                    "No supported primary or parallel templates for Observation {}".format(
                        observation_id
                    )
                )
                self.unusable_observations.append(observation_id)

        # Create seperate data object with unusable observations removed.
        self.supported_observations = self.program_data
        for observation_id in self.unusable_observations:
            self.supported_observations.pop(observation_id)

    def construct_exposure_dataframes(self):
        for observation_id in self.observation_number_list:
            exposure_table = self.program_data[observation_id]["exposures"]

            # Get V2 & V3 reference angles into exposure dataframe
            V2_ref, V3_ref = calculate_V2_V3_reference(exposure_table["AperName"])
            exposure_table["V2_ref"] = V2_ref
            exposure_table["V3_ref"] = V3_ref


            exposure_table = exposure_table.loc[exposure_table["apt_label"] != "BASE"]


def calculate_V2_V3_reference(apertures, instrument="NIRCAM"):
    """Use pysiaf to obtain V2, V3 reference coordinates

    Parameters
    ----------
    apertures: pd.Series
        A Pandas Series of strings containing aperture names
    instrumet: str
        JWST Instrument name to generate siaf with

    Returns
    -------
    V2Ref: pd.Series
        A Pandas Series containing the V2 reference angles
    V3Red: pd.Series
        A Pandas Series containing the V3 reference angles
    """

    siaf = pysiaf.Siaf(instrument)

    V2Ref = pd.Series(
        [siaf[aperture].V2Ref for aperture in apertures], index=apertures.index
    )
    V3Ref = pd.Series(
        [siaf[aperture].V3Ref for aperture in apertures], index=apertures.index
    )

    return V2Ref, V3Ref
