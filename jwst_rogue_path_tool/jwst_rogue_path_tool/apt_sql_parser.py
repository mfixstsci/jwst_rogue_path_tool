#!/usr/bin/env python

import pandas as pd
from astropy.table import Table

class AptSqlFile:
    """An sql file exported by APT."""

    def __init__(self, sqlfile):
        """Read data from sql file. Get list of tables names."""
        self.__sqlfile = sqlfile
        self.__sql = self.read_sql_file()
        self.tablenames = self.get_table_names()

    def build_aptsql_dataframe(self, tablename, show_table=False):
        """Create pandas dataframe from scraped apt sql data"""
        # Obtain metadata, create dataframe, apply to_numeric to convert all
        # numeric like columns from type object to int or float
        data = self.get_aptsql_metadata(tablename)
        df = pd.DataFrame(data)
        df = df.apply(pd.to_numeric, errors='ignore')

        # Show table in web browser
        if show_table:
            t = Table.from_pandas(df)
            t.show_in_browser()

        return df

    def get_aptsql_metadata(self, tablename):
        """Return a list of dictionaries with key/value pairs scraped from sql file."""
        prefix = "insert into " + tablename + " "
        rows = list()

        for line in self.__sql:
            if line[: len(prefix)] == prefix:
                keyval_str = line[len(prefix) :].strip()
                keystr, valstr = keyval_str.split("values")
                keys = [k.strip() for k in keystr[2:-2].split(",")]
                vals = [v.strip() for v in valstr[2:-2].split(",")]
                vals = [v.replace("'", '') for v in vals]   # Strip out the extra quotes from file that is parsed.
                keyval_dict = dict(zip(keys, vals))
                rows.append(keyval_dict)

        return rows

    def get_table_names(self):
        """Parse sql insert statements to determine table names."""

        prefix = "insert into "
        names = list()

        for line in self.__sql:
            if line[: len(prefix)] == prefix:
                names.append(line[len(prefix) : line.find("(")].strip())

        names = sorted(list(set(names)))
        names.remove("#AOK values")

        return names

    def read_sql_file(self):
        """Read sql file exported by APT. Strip trailing newlines."""

        sql = list()

        with open(self.__sqlfile, "r") as f:
            for line in f:
                sql.append(line.rstrip())

        return sql
