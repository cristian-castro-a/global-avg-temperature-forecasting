import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from typing import Dict, Tuple
from statsmodels.tsa.arima_model import ARIMA, ARIMAResultsWrapper

from utils.plotting import create_corr_plot
from utils.sdk_config import SDKConfig

logger = logging.getLogger(__name__)


def get_correlation_array(series: pd.Series, nlags: int, plot_type: str) -> np.array:
    """ To get acf and pacf values with 95% confidence intervals"""
    logger.info('Calculating correlation array')
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
    logger.info('p-value calculated')
    if p_value <= sig:
        print(f" {col_name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {col_name}e : P-Value = {p_value} => Non-stationary.")
    return p_value


def arima_model(data: pd.DataFrame, col_name: str, p: int, d: int, q: int, transform: str) -> ARIMAResultsWrapper:
    """Creates ARIMA model saves summary to text file"""
    logger.info(f'Fitting ARIMA model for p= {p}, q= {q} and d= {d}')
    model = ARIMA(data[col_name], order=(p, d, q))
    model_fit = model.fit(disp=0)

    # Save summary to text file
    file_name = 'arima_' + transform + '_' + str(p) + str(d) + str(q) + '.txt'
    path_to_results = SDKConfig().get_output_dir("ARIMA_summary") / file_name
    logger.info(f'Writing ARIMA summary to file at {path_to_results}')
    with open(path_to_results, 'w') as f:
        print(model_fit.summary(), file=f)

    return model_fit


def plot_auto_correlation(df: pd.DataFrame, col_name: str, nlags: int, transformer_name: str) -> None:
    # get auto-correlation arrays
    acf_array = get_correlation_array(df[col_name], nlags=nlags, plot_type='acf')
    pacf_array = get_correlation_array(df[col_name], nlags=nlags, plot_type='pacf')

    # plot ACF and PACF
    logger.info('Plotting ACF and PACF')
    path_to_results = SDKConfig().get_output_dir("ARIMA_corr_plots")
    create_corr_plot(corr_array=acf_array, title=transformer_name + '_ACF', path_to_results=path_to_results,
                     file_name=transformer_name + '_ACF.html')

    create_corr_plot(corr_array=pacf_array, title=transformer_name + '_PACF', path_to_results=path_to_results,
                     file_name=transformer_name + '_PACF.html')


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
        logger.info('Performing Box-Cox transformation')
        min_value = min(self.data[self.transform_column])

        # Adding scalar to make values positive
        self.data['gat_adj'] = self.data[self.transform_column] + abs(min_value) + scalar

        if lam is None:
            self.data['bc_gat'], lam = boxcox(self.data['gat_adj'])
        else:
            self.data['bc_gat'] = boxcox(self.data['gat_adj'], lmbda=lam)
        # Reverting scalar effects
        self.data['bc_gat'] = self.data['bc_gat'] - min_value - scalar
        self.data.rename(columns={'bc_gat': 'transformed_gat'}, inplace=True)
        return self.data, lam

    def log_transform(self, scalar: float) -> pd.DataFrame:
        """Does the log transformation on the data"""
        logger.info('Performing log transformation')
        trans_data, lmbda = self.box_cox_transform(scalar, lam=0.0)
        return trans_data

    def square_root_transform(self, scalar: float) -> pd.DataFrame:
        """Does square root transformation"""
        logger.info('Performing square root transformation')
        trans_data, lmbda = self.box_cox_transform(scalar, lam=0.5)
        return trans_data

    def cube_root_transform(self) -> pd.DataFrame:
        """Does cube root transformation"""
        logger.info('Performing cube root transformation')
        self.data['transformed_gat'] = np.cbrt(self.data[self.transform_column])
        return self.data
