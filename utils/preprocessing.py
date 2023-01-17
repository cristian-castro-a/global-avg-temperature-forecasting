from typing import Dict

import pandas as pd


def preprocess_co2_emissions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(by=['year']).sum().reset_index()[['year','co2']]
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['year','month','day']]))
    df.drop(['year','month','day'], axis=1, inplace=True)
    # To work with tonnes of co2 it is necessary a conversion factor of 3.664
    df['co2'] = df['co2']/3.664
    return df


def preprocess_dataframe(df_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        df_name: Name of the dataframe to preprocess
        df: Dataframe to preprocess
    Returns:
        Preprocessed dataframe
    """
    preprocessed_df = pd.DataFrame()
    if df_name == 'co2_emissions':
        preprocessed_df = preprocess_co2_emissions(df=df)
    return preprocessed_df


def preprocess_data(data_dict: Dict) -> Dict:
    """
    Parameters:
        data_dict: Dictionary with dataframes (each of which to be preprocessed differently)
    Returns:
        Dictionary of preprocessed dataframes
    """
    processed_data_dict = {}
    for df_name, df in data_dict.items():
        processed_data_dict[df_name] = preprocess_dataframe(df_name=df_name, df=df)
    return processed_data_dict
