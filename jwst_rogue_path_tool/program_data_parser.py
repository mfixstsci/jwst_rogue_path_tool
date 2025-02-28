"""This module contains all of the routines for parsing a JWST APT JSON output files

Authors
-------
    - Mario Gennaro
    - Mees Fix

Use
---
    Routines in this module can be imported as follows:

    >>> from jwst_rogue_path_tool.program_data_parser import aptJsonFile
    >>> filename = "/path/to/apt_json_file.json"
    >>> json = aptJsonFile(filename)
"""

from astropy.table import Table
import json
import pandas as pd


class aptJsonFile:
    """Read and parse JSON file generated from APT"""

    def __init__(self, json_file):
        """
        Parameters
        ----------
        json_file : str
            Name of APT JSON file
        """
        self.json_file = json_file
        self.data = self.read_json_file()
        self.tablenames = list(self.data.keys())

    def build_dataframe(self, tablename, show_table=False):
        """Create pandas dataframe from parsed APT JSON file

        Parameters
        ----------
        tablename : str
            Name of table to generate pandas dataframe from.

        show_table : bool
            Show table in web browser.
        """
        df = pd.DataFrame(self.data[tablename])
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df)

        # Show table in web browser
        if show_table:
            t = Table.from_pandas(df)
            t.show_in_browser()

        return df

    def read_json_file(self):
        """Read JSON file exported by APT"""

        with open(self.json_file, "r") as file:
            data = json.load(file)

        return data
