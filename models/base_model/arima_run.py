import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import math
from models.base_model.base_model_utils import get_correlation_array, adf_test, arima_model, DataTransformers, \
    plot_auto_correlation, diff_inv, mavg_inv
from utils.exploratory_data_analysis import get_eda_summary, plot_time_series
from utils.parsers import read_data
from utils.plotting import plot_lines_by
from utils.preprocessing import preprocess_data
from utils.sdk_config import SDKConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Finding best ARIMA model
    """

    logger.info('Reading data dict')

    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)
    processed_data_dict = preprocess_data(data_dict=data_dict, config=config)

    df_gat = processed_data_dict.get('global_temperature')
    lowest_aic = 10000
    lowest_model_fit = None
    best_model_name = ''

    logger.info('Looping on transformers')
    for transformer, name in config['transformers'].items():

        # transform dataframe
        logger.info('Transforming data')
        df_transformed = df_gat.copy()
        if transformer == 'sqrt':
            df_transformed = DataTransformers(df_transformed, 'global_temperature').square_root_transform(
                scalar=config['bxcx_scalar'])
        elif transformer == 'log':
            df_transformed = DataTransformers(df_transformed, 'global_temperature').log_transform(
                scalar=config['bxcx_scalar'])
        elif transformer == 'bxcx':
            df_transformed, lmbda = DataTransformers(df_transformed, 'global_temperature').box_cox_transform(
                scalar=config['bxcx_scalar'])
        elif transformer == 'cbrt':
            df_transformed = DataTransformers(df_transformed, 'global_temperature').cube_root_transform()
        elif transformer in ['mavg_1', 'mavg_2', 'mavg_3']:
            size = int(transformer[-1:])  # extract window size
            df_transformed = DataTransformers(df_transformed, 'global_temperature').moving_average_transform(
                window_size=size)
            df_transformed.dropna(inplace=True)
        else:
            df_transformed.rename(columns={'global_temperature': 'transformed_gat'}, inplace=True)

        # Correlation plots
        logger.info(f'Plotting correlation for {name}')
        nlags = math.floor((len(df_transformed.index) / 2) - 1)
        plot_auto_correlation(df=df_transformed, col_name='transformed_gat', nlags=nlags, transformer_name=name)

        # ADFuller Test
        logger.info(f'Performing ADFuller test for {name}')
        p_value = adf_test(data=df_transformed, col_name='transformed_gat', sig_value=config['sig_value'])

        # Modelling ARIMA
        ar_values = config['order']['p']
        diff_values = config['order']['d']
        ma_values = config['order']['q']
        if p_value > config['sig_value']:
            if 0 in diff_values: diff_values.remove(0)
        logger.info(f'Finding best ARIMA fit for {name}')
        for p in ar_values:
            for q in ma_values:
                for d in diff_values:
                    try:
                        model_fit = arima_model(data=df_transformed, col_name='transformed_gat', p=p, d=d, q=q,
                                                transform=name)
                        aic = model_fit.aic
                    except:
                        aic = lowest_aic
                        model_fit = lowest_model_fit
                    if aic < lowest_aic:
                        lowest_aic = aic
                        lowest_model_fit = model_fit
                        best_model_name = 'arima' + '_' + name + '_' + str(p) + str(d) + str(q) + '.pkl'
                        logger.info(f'AIC improved to {lowest_aic} with model name {best_model_name}')
        # Delete transformed dataframe
        del df_transformed

    # Create line plot visualisation for the actual and predicted values
    logger.info(f'Best model found!: {best_model_name}')
    df_gat_pred = df_gat.copy()
    diff_value = int(best_model_name[-6:-5])  # extract difference values

    # Convert moving averaged data to original scale
    if "Moving_Average" in best_model_name:
        # pre-processing
        logger.info(f'Converting Moving averaged data back to original scale')
        size = int(best_model_name[-9:-8])  # extract window size
        df_gat_pred.insert(1, 'pred_gat', lowest_model_fit.predict(dynamic=False))
        df_temp = df_gat_pred.copy()
        df_temp.drop(columns=['date', 'global_temperature'], inplace=True)
        df_temp.dropna(inplace=True)
        df_transformed = DataTransformers(df_gat_pred, 'global_temperature').moving_average_transform(
            window_size=size)

        # invert differencing data to original scale
        logger.info(f'Inverting differenced values for moving averaged data')
        if diff_value != 0:
            for i in range(diff_value + 1):
                temp_series = diff_inv(df_temp['pred_gat'].dropna(),
                                       df_transformed["transformed_gat"].iloc[size - 1])
        else:
            temp_series = df_temp['pred_gat'].dropna()
        df_gat_pred['temp_series'] = np.nan
        df_gat_pred['temp_series'][size - 1:] = temp_series

        # invert moving averaged data back to original scale
        s_diffed = size * df_gat_pred['temp_series'].diff()
        df_gat_pred = df_gat_pred.reset_index(drop=True)
        df_gat_pred['final_pred'] = mavg_inv(s=s_diffed, shift=size, df=df_gat_pred,
                                             col='global_temperature')
        logger.info(f'Moving averaged data inverted')

    # No transformation inversions done here since,
    # best model is either mavg or not_transformed based on BIC
    else:
        logger.info(f'Inverting differenced values for the model {best_model_name}')
        df_gat_pred.insert(diff_value, 'pred_gat', lowest_model_fit.predict(dynamic=False))
        if diff_value != 0:
            for i in range(diff_value + 1):
                df_gat_pred['final_pred'] = diff_inv(df_gat_pred['pred_gat'].dropna(),
                                                     df_gat_pred["global_temperature"].iloc[0])
        else:
            df_gat_pred['final_pred'] = df_gat_pred['pred_gat']

    # Plot the graph comparing original and predicted
    logger.info(f'Plotting predictions for {best_model_name}')
    path_to_results = SDKConfig().get_output_dir("model_forecasts")
    plot_lines_by(data=df_gat_pred, plot_x='date', plot_y=['global_temperature', 'final_pred'],
                  path_to_results=path_to_results, file_name=f"{best_model_name}.html",
                  x_title='date', y_title=f"Actual and {best_model_name} Prediction")

    # Saving best ARIMA model
    path_to_best_arima = SDKConfig().get_best_arima_dir() / best_model_name
    logger.info(f'Saving best ARIMA model to path {path_to_best_arima}')
    lowest_model_fit.save(path_to_best_arima)

    logger.info('Model Saved, exiting!')


if __name__ == '__main__':
    main()
