import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from build_uni_lstm import univariate_lstm
from utils.parsers import read_data
from utils.preprocessing import preprocess_data, Windowing, PredictorScaler
from utils.sdk_config import SDKConfig
from utils.plotting import plot_lines_by

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Univariate LSTM Model
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

    # Get Sequences of Data
    sequences_generator = Windowing(df=df_scaled, mode='Univariate')
    X, y = sequences_generator.get_input_sequences(on_column='global_temperature')

    # Create LSTM Model Based on Config File
    model = univariate_lstm(X_train=X, config=config)

    model.compile(
        loss = config.univariate_lstm_model.loss,
        optimizer=config.univariate_lstm_model.optimizer,
        metrics=config.univariate_lstm_model.metrics
    )

    history = model.fit(
        X,
        y,
        validation_split=config.univariate_lstm_model.test_size_split,
        epochs=config.univariate_lstm_model.epochs,
        batch_size=1
    )

    # Save Model
    path_to_model = SDKConfig().get_output_dir('univar_lstm_model') / 'model_4'
    model.save(path_to_model)

    # Plot Training and Validation Losses
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    loss_df = pd.DataFrame(
        {'epochs': [idx for idx in range(config.univariate_lstm_model.epochs)], 'loss': loss, 'val_loss': val_loss}
    )

    path_to_results = SDKConfig().get_output_dir("plots_univar_lstm_model_results(model_4)")
    plot_lines_by(data=loss_df, plot_x='epochs', plot_y=['loss', 'val_loss'],
                  path_to_results=path_to_results, file_name="loss_plots.html",
                  x_title='epochs', y_title="Training Losses")

    # Plot Errors
    mse = history.history['mean_squared_error']
    mae = history.history['mean_absolute_error']

    error_df = pd.DataFrame(
        {'epochs': [idx for idx in range(config.univariate_lstm_model.epochs)], 'mse': mse, 'mae': mae}
    )

    path_to_results = SDKConfig().get_output_dir("plots_univar_lstm_model_results(model_4)")
    plot_lines_by(data=error_df, plot_x='epochs', plot_y=['mse', 'mae'],
                  path_to_results=path_to_results, file_name="error_plots.html",
                  x_title='epochs', y_title="Training Errors")

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform the predictions to get the actual values
    predictions = scaler.inverse_transform(predictions)

    # Create line plot visualisation for the actual and predicted values
    df_lstm_res = processed_data_dict.get('global_temperature')
    df_lstm_res = df_lstm_res.iloc[5:].assign(predictions=predictions)

    path_to_results = SDKConfig().get_output_dir("plots_univar_lstm_model_results(model_4)")
    plot_lines_by(data=df_lstm_res, plot_x='date', plot_y=['global_temperature', 'predictions'],
                  path_to_results=path_to_results, file_name="univar_lstm.html",
                  x_title='date', y_title="Actual differenced and univar LSTM Prediction")


if __name__ == '__main__':
    main()