import logging

import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path

from build_uni_lstm import univariate_lstm
from utils.parsers import read_data
from utils.preprocessing import preprocess_data, Windowing, PredictorScaler
from utils.sdk_config import SDKConfig
from utils.plotting import plot_lstm_results

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
    scaled_data = scaler.transform(variable_values=processed_data_dict['global_temperature']['global_temperature'])
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
        validation_split=0.1,
        epochs=5,
        batch_size=1
    )

    # Get predictions
    path_to_results = SDKConfig().get_output_dir("plots_model_results")
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    plot_lstm_results(history, processed_data_dict['global_temperature']['global_temperature'][-len(predictions):],
                      predictions.flatten(), path_to_results, 'lstm_results.html')


if __name__ == '__main__':
    main()