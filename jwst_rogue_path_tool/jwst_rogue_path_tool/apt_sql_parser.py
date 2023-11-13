#!/usr/bin/env python

from astropy.table import Table


class AptSqlFile:
    """An sql file exported by APT."""

    def __init__(self, sqlfile):
        """Read data from sql file. Get list of tables names."""
        self.__sqlfile = sqlfile
        self.__sql = self.read_sql_file()
        self.tablenames = self.get_table_names()

    def build_aptsql_database_table(self, tablename, browser=False):
        """Construct astropy table from sql insert statements.
        For the 'exposures' table, discard rows with apt_label == 'BASE'.
        Convert column data type to integer or float, where possible.
        Strip beginning and ending single quote from strings.
        """
        rows = self.get_aptsql_metadata(tablename)
        keys = self.get_aptsql_table_column_names(rows)

        if len(keys) == 0:
            raise Exception("no '" + tablename + "' table in " + self.__sqlfile)

        table = self.build_aptsql_metadata_table(rows, keys)

        if browser:
            self.show_table_in_browser(table)

        return table

    def build_aptsql_metadata_table(self, rows, keys):
        """Build an astropy table with sql metadata and column names.

        >>> rows = AptSqlFile.get_aptsql_metadata(tablename)
        >>> keys = AptSqlFile.get_aptsql_table_column_names(rows)
        >>> AptSqlFile.build_aptsql_metadata_table(rows, keys)

        Returns
        -------
        table: astropy.Table.table
            An astropy table with database apt metadata
        """
        table = Table()

        for key in keys:
            col = list()

            for row in rows:
                col.append(row.get(key, ""))

            try:
                col = [int(x) for x in col]
            except ValueError:
                try:
                    col = [float(x) for x in col]
                except ValueError:
                    col = [
                        x[1:-1] if x.startswith("'") and x.endswith("'") else x
                        for x in col
                    ]

            table[key] = col

        return table

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
                keyval_dict = dict(zip(keys, vals))
                rows.append(keyval_dict)

        return rows

    def get_aptsql_table_column_names(self, rows):
        """Return a list of key names from dictionary entries.

        rows: list-like
            A list of dictionaries

        Returns
        -------
        keys: list-like
            A comprehensive list of key names from rows object
        """
        keys = set()

        for row in rows:
            keys = keys.union(row.keys())

        keys = sorted(list(keys))

        return keys

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

    def show_table_in_browser(self, table):
        """Diplay copy of astropy table in a browser window.
        Convert underscores to spaces in column headers to allow wrapping.
        """
        out = table.copy(copy_data=False)

        for key in out.keys():
            newkey = key.replace("_", " ")
            if newkey != key:
                out.rename_column(key, newkey)

        out.show_in_browser(jsviewer=True, show_row_index=False)
