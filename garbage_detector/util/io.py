import csv
import os
import time
from typing import List

import pandas as pd
from IPython.display import display


def get_project_root_dir():
    """Returns the absolute path of the root directory of this project
    This is to make sure that files are always stored in the correct directory.

    Returns
    -------
    str

    """
    script_path = os.path.abspath(__file__)
    seperator = ('\\garbage_detector' if os.name ==
                 'nt' else '/garbage_detector')
    project_root = script_path.split(seperator, 1)[0]
    return project_root


class CSVWriter:
    """
    A helper class for writing csv files.
    If the CSV file already exist, a new file with the same name
    but the suffix "_{unix_time_s} attached to it, e.g.
    benchmark_1679082010.csv

    Attributes
    --------
    csv_path: str
        Path to the CSV file which the writer opens and writes to.

    csv_file: io.TextIOWrapper
        Pointer to the actual CSV file

    writer: csv.DictWriter
        Responsable for writing to the CSV file.

    """

    def __init__(self, csv_path: str, headers):
        """
        Parameters
        ----------
        csv_path : str
            Relative path to the csv file
        headers: dict
            The headers, the CSV file must contain
        """
        csv_abs_path = os.path.join(get_project_root_dir(), csv_path)
        os.makedirs(os.path.dirname(csv_abs_path), exist_ok=True)
        self.csv_path = self.if_exists_create_new(csv_abs_path)
        self.csv_file = open(self.csv_path, 'a+', newline='')
        self.writer: csv.DictWriter = csv.DictWriter(
            self.csv_file, fieldnames=headers
        )

    def if_exists_create_new(self, csv_path: str):
        """Checks if file with name already exists.
        Checks whether the file with the name already exists.
        If this is the case, a new file is created with an additional suffix
        in the name and its path is returned. Otherwise, csv_path is returned,
        so that it is used directly to create a new file.

        Parameters
        ----------
        csv_path: int
            Absolute path to the CSV file.

        Returns
        -------
        str
        """
        if os.path.exists(csv_path):

            suffix: str = str(int(time.time()))

            new_path = os.path.splitext(csv_path)[0] + '_' + suffix + '.csv'
            open(new_path, 'w+').close()

            return new_path
        else:
            return csv_path

    def write_header(self):
        """Writes header to CSV file

        Returns
        -------
        None
        """
        self.writer.writeheader()
        self.csv_file.flush()

    def write_rows(self, rows: List):
        """Writes list of rows to CSV file

        Parameters
        ----------
        rows: list[str]

        Returns
        -------
        None
        """
        for row in rows:
            self.writer.writerow(row)
            self.csv_file.flush()

    def close(self):
        """Closes CSV file

        Returns
        -------
        None
        """
        self.csv_file.close()


class CSVReader:
    """
    A helper class for reading CSV files.
    Non existing CSV files are ignored.
    Content of CSV file is displayed as HTML table in the jupyter Notebook

    Attributes
    --------
    csv_path: str
        Absolute Path to the CSV file which the reader opens to read from.
    """

    def __init__(self, csv_path: str):
        """
        Parameters
        ----------
        csv_path : str
            Relative path to the csv file
        """
        self.csv_path = os.path.join(get_project_root_dir(), csv_path)

    def __call__(self):
        """Prints content of CSV file as HTML in jupyter notebook"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            display(df)
