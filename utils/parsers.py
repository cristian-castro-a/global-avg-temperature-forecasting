from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class CSVParser:
    """
    Class for parsing csv data.

    Expected parameters:
    - csv_file: name of raw data file stored in data_path
    """
    csv_file: str
    data_path: str

    def __repr__(self):
        resp = f"Data handler for {self.csv_file}"
        return resp

    def _check_existence(self):
        data_path = Path(self.data_path)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Folder not found in {data_path}")
        file_path = data_path.joinpath(self.csv_file)
        if not file_path.is_file():
            raise FileNotFoundError(f"\nThere is no file named {self.csv_file} in {data_path}")
        return file_path

    def load_raw_data(self):
        file_path = self._check_existence()
        return pd.read_csv(file_path)


@dataclass
class CountryPoverty(CSVParser):
    """
    Class to store poverty data per country per ppp_version
    """
    country_name: str
    ppp_version: int

    def load_country_data(self):
        df = self.load_raw_data()
        return df[(df['country'] == self.country_name) & (df['ppp_version'] == self.ppp_version)]