import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class Windowing:
    """
    Class to create fixed length sequences to train a LSTM Model
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_input_sequences(self, on_column: str, window: int = 5) -> Tuple[np.array, np.array]:
        assert isinstance(window, int), f" 'Window' is expected to be an integer, but got {type(window)}."
        assert isinstance(on_column, str), f" 'On_column' is expected to be a string, but got {type(on_column)}."
        assert on_column in self.df.columns, f" {on_column} is not a column in the dataframe."

        df_as_np = self.df[on_column].to_numpy()

        X = []
        y = []

        for i in range(len(df_as_np)-window):
            X.append([[a] for a in df_as_np[i:i+window]])
            y.append(df_as_np[i+window])

        return np.array(X), np.array(y)


def preprocess_co2_emissions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(by=['year']).sum().reset_index()[['year','co2']]
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['year','month','day']]))
    df.drop(['year','month','day'], axis=1, inplace=True)
    # To work with tonnes of co2 it is necessary a conversion factor of 3.664
    df['co2'] = df['co2']/3.664
    df.rename(columns={'date': 'date', 'co2': 'co2_emissions'}, inplace=True)
    return df


def preprocess_global_temperature(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['Year','month','day']]))
    df.drop(['Year','month','day', 'Lowess(5)'], axis=1, inplace=True)
    df.rename(columns={'date': 'date', 'No_Smoothing': 'global_temperature'}, inplace=True)
    return df



def preprocess_ocean_warming(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['Year','month','day']]))
    df.drop([col for col in df.columns if col not in ('date', 'NOAA')], axis=1, inplace=True)
    df.rename(columns={'index': 'date', 'NOAA': 'ocean_warming'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df


def preprocess_world_employment(df:pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(by=['TIME']).mean().reset_index()
    df = df[df['TIME'] > 1964]
    df = df.rename(columns = {'TIME':'year'})
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['year','month','day']]))
    df.drop(['year','month','day'], axis=1, inplace=True)
    df.rename(columns= {'Value':'world_employment_rate'}, inplace=True)
    return df

def preprocess_energy_substitution(df:pd.DataFrame) -> pd.DataFrame:
    column_list = ['Other renewables (TWh, substituted energy)',
                   'Biofuels (TWh, substituted energy)',
                    'Solar (TWh, substituted energy)',
                   'Wind (TWh, substituted energy)',
                   'Hydropower (TWh, substituted energy)',
                   'Nuclear (TWh, substituted energy)',
                   'Gas (TWh, substituted energy)',
                   'Oil (TWh, substituted energy)',
                   'Coal (TWh, substituted energy)',
                   'Traditional biomass (TWh, substituted energy)']
    df['global_energy_substitution'] = df[column_list].sum(axis=1)
    df = df[['Year', 'global_energy_substitution']]
    df = df[df['Year'] > 1964]
    df.insert(loc=1, column='month', value=12)
    df.insert(loc=2, column='day', value=31)
    df.insert(loc=0, column='date', value=pd.to_datetime(df[['Year','month','day']]))
    df.drop(['Year','month','day'], axis=1, inplace=True)
    return df

def preprocess_world_population(df:pd.DataFrame) -> pd.DataFrame:
    df.reset_index(inplace= True)
    df.rename(columns={' Population':'world_population'}, inplace=True)
    df.date = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year > 1964]
    df = df[['date','world_population']]
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
    if df_name == 'global_temperature':
        preprocessed_df = preprocess_global_temperature(df=df)
    if df_name == 'ocean_warming':
        preprocessed_df = preprocess_ocean_warming(df=df)
    if df_name == 'world_population':
        preprocessed_df = preprocess_world_population(df=df)
    if df_name == 'world_employment_rate':
        preprocessed_df = preprocess_world_employment(df=df)
    if df_name == 'global_energy_substitution':
        preprocessed_df = preprocess_energy_substitution(df=df)
    return preprocessed_df


def preprocess_data(data_dict: Dict) -> Dict:
    """
    Parameters:
        data_dict: Dictionary with dataframes (each of which to be preprocessed differently)
    Returns:
        Dictionary of preprocessed dataframes
    """
    processed_data_dict = {}
    logger.info('Preprocessing dataframes')
    for df_name, df in data_dict.items():
        logger.info(f'Preprocessing dataframe: {df_name}')
        processed_data_dict[df_name] = preprocess_dataframe(df_name=df_name, df=df)
    return processed_data_dict
