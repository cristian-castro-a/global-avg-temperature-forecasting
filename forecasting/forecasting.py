import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig

from utils.parsers import read_data
from utils.plotting import plot_lines_by
from utils.preprocessing import preprocess_data, Windowing, PredictorScaler
from utils.sdk_config import SDKConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def run_forecast(config: DictConfig) -> None:
    """
    Forecasting using best LSTM Model
    """
    logger.info('Setting working directories for the project')
    SDKConfig().set_working_dirs()

    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)
    processed_data_dict = preprocess_data(data_dict=data_dict, config=config)

    # Get Scaled Data
    scaler = PredictorScaler(scaler=config.univariate_lstm_model.scaler,
                             variable_values=processed_data_dict['global_temperature']['global_temperature'],
                             feature_range=(0,1)).fit()
    scaled_data = scaler.transform(pd.DataFrame(processed_data_dict['global_temperature']['global_temperature']))
    df_scaled = pd.DataFrame(scaled_data, columns=['global_temperature'])

    # Get Sequences of Real Data
    sequences_generator = Windowing(df=df_scaled, mode='Univariate')
    X_real, y_real = sequences_generator.get_input_sequences(on_column='global_temperature')

    # Load Best Model
    path_to_best_model = Path(f"../{config['best_model']}")
    best_model = tf.keras.models.load_model(path_to_best_model)

    # Get Predictions of Best Model
    X_predicted = best_model.predict(X_real)

    # Forecasting: The model starts at the last 5 points in the time series
    X_forecast = np.array([X_predicted[-5:]])
    X_forecast_predictions = []

    # Let's forecast the next 10 years: 2022 - 2031
    for iteration in range(1,11):
        prediction = best_model.predict(X_forecast)
        X_forecast_predictions.append([prediction[-1][0]])

        to_append = X_forecast.tolist()
        to_append = to_append[iteration-1][-4:]
        to_append.append([prediction[-1][0]])

        X_forecast = X_forecast.tolist()
        X_forecast.append(to_append)
        X_forecast = np.array(X_forecast)

    X_forecast_total = np.append(X_predicted, np.array(X_forecast_predictions)).reshape(-1,1)
    X_forecast_total = scaler.inverse_transform(X_forecast_total)

    forecast_df = pd.DataFrame({
            'date': pd.date_range(start='1970-12-31', end='2029-12-31', freq='Y'),
            'global_temperature_forecasted': X_forecast_total.reshape(-1).tolist()
        })
    real_df = processed_data_dict.get('global_temperature')

    total_df = pd.merge(forecast_df, real_df, how="left", on=['date'])

    path_to_results = SDKConfig().get_output_dir("plots_forecasting_results")
    plot_lines_by(data=total_df, plot_x='date', plot_y=['global_temperature', 'global_temperature_forecasted'],
                  path_to_results=path_to_results, file_name="forecast.html",
                  x_title='Date (YYYY-MM-DD)', y_title="Global Avg. Temperature (°C)")


if __name__ == '__main__':
    run_forecast()