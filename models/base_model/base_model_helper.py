import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def get_correlation_array(series: pd.Series, nlags: int, plot_type: str) -> np.array:
    """ To get acf and pacf values with 95% confidence intervals"""
    if plot_type == 'pacf':
        corr_array = pacf(series.dropna(), nlags=nlags, alpha=0.05)
    else:
        corr_array = acf(series.dropna(), alpha=0.05)
    return corr_array


def adf_test(data: pd.DataFrame, col_name: str, sig_value: float) -> float:
    """ADFuller test to check the stationarity of the data"""
    res = adfuller(data[col_name], autolag='AIC')
    p_value = round(res[1], 3)
    sig = sig_value
    if p_value <= sig:
        print(f" {col_name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {col_name}e : P-Value = {p_value} => Non-stationary.")
    return p_value


class DataTransformers:
    """To do transformations on the data to make it stationary"""

    def __init__(self, df: pd.DataFrame, column: str):
        self.data = df
        self.transform_column = column

    def box_cox_transform(self, scalar: float, lam=None) -> Tuple[pd.DataFrame, float]:
        """
        Does the transformation most appropriate to the data
        but only works with positive values so a scalar value
        must be passed to convert negative values into positive
        """
        min_value = min(self.data[self.transform_column])

        # Adding scalar to make values positive
        self.data['gat_adj'] = self.data[self.transform_column] + abs(min_value) + scalar

        if lam is None:
            self.data['bc_gat'], lam = boxcox(self.data['gat_adj'])
        else:
            self.data['bc_gat'] = boxcox(self.data['gat_adj'], lmbda=lam)
        # Reverting scalar effects
        self.data['bc_gat'] = self.data['bc_gat'] - min_value - scalar
        return self.data, lam

    def log_transform(self, scalar: float) -> pd.DataFrame:
        """Does the log transformation on the data"""
        trans_data, lmbda = self.box_cox_transform(scalar, lam=0.0)
        trans_data.rename(columns={'bc_gat': 'log_gat'}, inplace=True)
        return trans_data

    def square_root_transform(self, scalar: float) -> pd.DataFrame:
        """Does square root transformation"""
        trans_data, lmbda = self.box_cox_transform(scalar, lam=0.5)
        trans_data.rename(columns={'bc_gat': 'sqrt_gat'}, inplace=True)
        return trans_data

    def cube_root_transform(self) -> pd.DataFrame:
        """Does cube root transformation"""
        self.data['cbrt_gat'] = np.cbrt(self.data[self.transform_column])
        return self.data
