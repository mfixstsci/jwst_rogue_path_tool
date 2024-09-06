"""This module contains all of the routines for parsing a JWST APT SQL file.

Authors
-------
    - Mario Gennaro
    - Mees Fix

Use
---
    Routines in this module can be imported as follows:

    >>> from jwst_rogue_path_tool.apt_sql_parser import AptSqlFile
    >>> filename = "/path/to/sql_apt_file.sql"
    >>> sql = AptSqlFile(filename)
"""

import pandas as pd
from astropy.table import Table


class aptSqlFile:
    """Read and parse SQL file generated from APT"""

    def __init__(self, sqlfile):
        """
        Parameters
        ----------
        sqlfile : str
            Name of APT SQL file (filename.sql)
        """
        self.__sqlfile = sqlfile
        self.__sql = self.read_sql_file()
        self.tablenames = self.get_table_names()

    def build_aptsql_dataframe(self, tablename, show_table=False):
        """Create pandas dataframe from parsed APT file

        Parameters
        ----------
        tablename : str
            Name of table to generate pandas dataframe from.

        show_table : bool
            Show table in web browser.
        """
        # Obtain metadata, all data read from sql file are strings.
        # This changes all non-numerics to nans then uses the nan
        # values as a way to mask and back fill strings.
        data = self.get_aptsql_metadata(tablename)
        df = pd.DataFrame(data)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df)

        # Show table in web browser
        if show_table:
            t = Table.from_pandas(df)
            t.show_in_browser()

        return df

    def get_aptsql_metadata(self, tablename):
        """Return a list of dictionaries with key/value pairs parsed from sql file.

        Parameters
        ----------
        tablename : str
            Table name that is used to parse out data from SQL file.
        """
        prefix = "insert into " + tablename + " "
        rows = list()

        for line in self.__sql:
            if line[: len(prefix)] == prefix:
                keyval_str = line[len(prefix) :].strip()
                keystr, valstr = keyval_str.split("values")
                keys = [k.strip() for k in keystr[2:-2].split(",")]
                vals = [v.strip() for v in valstr[2:-2].split(",")]
                vals = [
                    v.replace("'", "") for v in vals
                ]  # Strip out the extra quotes from file that is parsed.
                keyval_dict = dict(zip(keys, vals))
                rows.append(keyval_dict)

        return rows

    def get_table_names(self):
        """Parse SQL insert statements to determine table names."""

        prefix = "insert into "
        names = list()

        for line in self.__sql:
            if line[: len(prefix)] == prefix:
                names.append(line[len(prefix) : line.find("(")].strip())

        names = sorted(list(set(names)))
        names.remove("#AOK values")

        return names

    def read_sql_file(self):
        """Read SQL file exported by APT. Strip trailing newlines."""

        sql = list()

        with open(self.__sqlfile, "r") as f:
            for line in f:
                sql.append(line.rstrip())

        return sql
