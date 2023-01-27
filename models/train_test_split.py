from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from utils.exploratory_data_analysis import check_correct_dates


def split_dataframe_dict(processed_data_dict: Dict, test_size: float) -> Tuple[Dict, Dict]:
    """
    Splits a dictionary of dataframes into training and test sets
    Parameters:
        processed_data_dict: Dictionary of dataframes to be split
        test_size: Float between 0 and 1, representing the proportion of the data to be used for testing
    Returns:
        Tuple of two dictionaries, containing the training and test sets for each dataframe in the input dictionary
    """

    checked_data_dict = check_correct_dates(processed_data_dict=processed_data_dict)

    train_data_dict = {}
    test_data_dict = {}
    for df_name, df in checked_data_dict.items():
        train, test = train_test_split(df, test_size=test_size)
        train_data_dict[df_name] = train
        test_data_dict[df_name] = test
    return train_data_dict, test_data_dict
