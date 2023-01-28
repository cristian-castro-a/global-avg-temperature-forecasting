from typing import Dict

import pandas as pd
from utils.plotting import plot_lines_by
from pathlib import Path


def check_correct_structures(processed_data_dict: Dict) -> Dict:
    """
    Parameters:
        processed_data_dict: Dictionary of previously preprocessed time series data
    Returns:
        Same dictionary after checking correct structures (Only two columns per dataframe, where one of them is 'date')
    """
    for df_name, df in processed_data_dict.items():
        assert len(df.columns) == 2, f"The {df_name} dataframe has more than two columns." \
                                     f"Only two columns are allowed: 'date' and value of the predictor."
        assert 'date' in df.columns, f"One column of the dataframe {df_name} must be named 'date'."
        assert df['date'].dtype == 'datetime64[ns]', f"Column 'date' of {df_name} should be datetime64 type."
    return processed_data_dict


def check_correct_dates(processed_data_dict: Dict) -> Dict:
    """
    Parameters:
        processed_data_dict: Dictionary of previously preprocessed time series data
    Returns:
        Same dictionary after checking correct structures (Only two columns per dataframe, where one of them is 'date')
    """

    data_dict_relevant_years = check_correct_structures(processed_data_dict=processed_data_dict)

    for df_name, df in data_dict_relevant_years.items():
        assert len(df.columns) == 2, f"The {df_name} dataframe has more than two columns." \
                                   f"Only two columns are allowed: 'date' and value of the predictor."
        assert 'date' in df.columns, f"One column of the dataframe {df_name} must be named 'date'."
        assert df['date'].dtype == 'datetime64[ns]', f"Column 'date' of {df_name} should be datetime64 type."
        assert df['date'].min() == '1965-12-31', f"The first date of {df_name} should be 1965-12-31."
        assert df['date'].max() == '2019-12-31', f"The last date of {df_name} should be 2019-12-31."
    return data_dict_relevant_years


def get_eda_summary(processed_data_dict: Dict) -> pd.DataFrame:
    """
    Parameters:
        processed_data_dict: Dictionary of previously preprocessed time series data
    Returns:
        A dataframe with a summary of an EDA
    """
    checked_data_dict = check_correct_structures(processed_data_dict=processed_data_dict)

    columns = ['min_date', 'max_date', 'time_between_measures', 'count', 'mean', 'min', 'max']
    summary = {}

    for df_name, df in checked_data_dict.items():
        predictor_name = [col for col in df.columns if 'date' not in col][0]
        results = [
            df['date'].min(),
            df['date'].max(),
            df['date'].diff().mean(),
            df[predictor_name].count(),
            df[predictor_name].mean(),
            df[predictor_name].min(),
            df[predictor_name].max()
        ]
        summary[df_name] = results

    return pd.DataFrame.from_dict(summary, orient='index', columns=columns)


def plot_time_series(processed_data_dict: Dict, path_to_results: Path):
    """
    Parameters:
        processed_data_dict: Dictionary of previously preprocessed time series data
        path_to_results: the path where to store the results in .html
    Returns:
        None but plots the times series
    """
    checked_data_dict = check_correct_structures(processed_data_dict=processed_data_dict)

    # Iterate over the dictionary
    for df_name, df in checked_data_dict.items():
        file_name = df_name + ".html"
        plot_lines_by(data=df, plot_x='date', plot_y=df_name, path_to_results=path_to_results,
                      file_name=file_name, x_title='date', y_title=df_name)
