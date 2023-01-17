import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


@dataclass
class CSVParser:
    data_file_path: Path

    def _check_existence(self):
        data_dir = self.data_file_path.parent
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {data_dir}")
        if not self.data_file_path.is_file():
            raise FileNotFoundError(f"\nThere is no file named {self.data_file_path.name} in {data_dir}")
        return self.data_file_path

    def load_raw_data(self):
        file_path = self._check_existence()
        return pd.read_csv(file_path)


@dataclass
class JSONParser:
    data_file_path: Path

    def _check_existence(self):
        data_dir = self.data_file_path.parent
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {data_dir}")
        if not self.data_file_path.is_file():
            raise FileNotFoundError(f"\nThere is no file named {self.data_file_path.name} in {data_dir}")
        return self.data_file_path

    def load_raw_data(self):
        file_path = self._check_existence()
        with open(file_path) as json_file:
            json_dict = json.load(json_file)
        return pd.DataFrame.from_dict(json_dict, orient='index', columns=['value'])


@dataclass
class TXTParser:
    data_file_path: Path

    def _check_existence(self):
        data_dir = self.data_file_path.parent
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {data_dir}")
        if not self.data_file_path.is_file():
            raise FileNotFoundError(f"\nThere is no file named {self.data_file_path.name} in {data_dir}")
        return self.data_file_path

    def load_raw_data(self):
        file_path = self._check_existence()
        return pd.read_csv(file_path, delim_whitespace=True)


def read_data(data_dir: Path, config: DictConfig) -> Dict:
    """
    Parameters:
        data_dir: Path to data folder with raw data
        config: Config file
    Returns:
        dictionary of dataframe, each representing each file in data_dir
    """
    logger.info("Loading raw data")

    data_dict = {}

    for df, dataset in config['datafiles'].items():
        file_path = data_dir / dataset
        if dataset.endswith('.csv'):
            parser = CSVParser(data_file_path=file_path)
        elif dataset.endswith('.txt'):
            parser = TXTParser(data_file_path=file_path)
        elif dataset.endswith('.json'):
            parser = JSONParser(data_file_path=file_path)
        else:
            raise FileNotFoundError(f"Extension of file {dataset} not known.")
        data_dict[df] = parser.load_raw_data()

    return data_dict