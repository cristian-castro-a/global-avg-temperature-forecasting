import logging
from pathlib import Path

import hydra
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig

from utils.parsers import read_data
from utils.plotting import plot_lines_by
from utils.preprocessing import preprocess_data, Windowing, PredictorScaler
from utils.sdk_config import SDKConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def run_arima_vs_lstm(config: DictConfig) -> None:
    """
    ARIMA vs LSTM
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
    X_predicted = scaler.inverse_transform(X_predicted)

    lstm_df = pd.DataFrame({
            'date': pd.date_range(start='1970-12-31', end='2019-12-31', freq='Y'),
            'global_temperature_lstm': X_predicted.reshape(-1).tolist()
        })
    real_df = processed_data_dict.get('global_temperature')

    arima_df = pd.read_csv('preds.csv')
    arima_df['date'] = pd.to_datetime(arima_df['date'])
    arima_df = arima_df[['date', 'final_pred']]
    arima_df = arima_df.rename(columns={'date':'date', 'final_pred': 'global_temperature_arima'})

    total_df = pd.merge(lstm_df, arima_df, how="left", on=['date'])
    total_df = pd.merge(total_df, real_df, how="left", on=['date'])

    path_to_results = SDKConfig().get_output_dir("arima_vs_lstm_plots")
    plot_lines_by(data=total_df, plot_x='date', plot_y=['global_temperature', 'global_temperature_lstm', 'global_temperature_arima'],
                  path_to_results=path_to_results, file_name="arima_vs_lstm_plot.html",
                  x_title='Date (YYYY-MM-DD)', y_title="Global Avg. Temperature (°C)")


if __name__ == '__main__':
    run_arima_vs_lstm()