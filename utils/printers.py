import pandas as pd
from pathlib import Path


def print_dataframe_descriptive_statistics(data: pd.DataFrame, file_name: str, path_to_results: str) -> None:
    """
    Prints the descriptive statistics of a dataframe
    :param data: dataframe to obtain the descriptive statistics
    :param path_to_results: path to print the results
    :return: prints a .csv with the descriptive statistics
    """
    path_to_results = Path(path_to_results)

    if not path_to_results.is_dir():
        raise Exception(f"There is no directory named '{path_to_results.name}' in: '{path_to_results.parent}'")

    path_to_file = path_to_results.joinpath(file_name)

    descriptive_statistics = data.describe(percentiles=[0.1,0.25,0.5,0.75,0.9])

    descriptive_statistics.to_csv(path_to_file, index=False)