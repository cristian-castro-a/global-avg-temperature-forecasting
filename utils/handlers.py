from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class HandlingData:
    """
    Class for handling data.

    Expected parameters:
    - csv_file: name of raw data file stored in ../data
    """
    csv_file: str
    data_path: str

    def __repr__(self):
        resp = f"Data handler for {self.csv_file}"
        return resp

    def _check_existence(self):
        data_path = Path(self.data_path)
        if not data_path.is_dir():
            raise Exception(f"Folder not found in {data_path}")

        file_path = data_path.joinpath(self.csv_file)

        if not file_path.is_file():
            raise Exception(f"\nThere is no file named {self.csv_file} in {self.data_path}")

        return file_path

    def _load_raw_data(self):
        file_path = self._check_existence()
        return pd.read_csv(file_path)

    def load_raw_data(self):
        return self._load_raw_data()


@dataclass
class CountryPoverty(HandlingData):
    ppp_version: int
