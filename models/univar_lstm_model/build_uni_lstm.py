import tensorflow as tf
from typing import Dict, Tuple
import numpy as np
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_variables_from_config(config: DictConfig) -> Tuple[int, int, Dict, str]:
    nr_lstm_layers = config['univariate_lstm_model']['nr_lstm_layers']
    nr_dense_layers = config['univariate_lstm_model']['nr_dense_layers']
    units_per_layer = config['univariate_lstm_model']['units_per_layer']
    dense_layer_activation = config['univariate_lstm_model']['dense_layer_activation']

    assert nr_lstm_layers >= 1, "Number of LSTM layers must be at least 1."
    assert nr_dense_layers >= 1, "Number of Dense layers must be at least 1."
    assert nr_lstm_layers + nr_dense_layers == len(units_per_layer.keys()), "Bad specification of units per layer in " \
                                                                            "config file. Please check."
    assert dense_layer_activation in ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
                                      'hard_sigmoid',
                                      'linear'], f"{dense_layer_activation} is not valid activation function in Keras"

    return nr_lstm_layers, nr_dense_layers, units_per_layer, dense_layer_activation


def univariate_lstm(X_train: np.ndarray, config: DictConfig) -> tf.keras.models.Sequential:
    """
    Builds the architecture for a univariate LSTM model

    Parameters:
        X_train: ndarray for predictors
        config: Configuration file
    Return:
        Sequential LSTM Model
    """
    nr_lstm_layers, nr_dense_layers, units_per_layer, dense_layer_activation = get_variables_from_config(config=config)

    logger.info("Building sequential LSTM model")
    model = tf.keras.models.Sequential()

    # LSTM Layers
    for layer_nr in range(1, nr_lstm_layers + 1):
        layer_name = f"LSTM_{layer_nr}"
        nodes_layer = units_per_layer[layer_name]

        if layer_nr == 1:
            logger.info(f"Building sequential LSTM model - {layer_name}")
            model.add(
                tf.keras.layers.LSTM(
                    nodes_layer,
                    return_sequences=True,
                    input_shape=(X_train.shape[1], 1),
                    name=layer_name
                )
            )
        elif layer_nr == nr_lstm_layers:
            logger.info(f"Building sequential LSTM model - {layer_name}")
            model.add(
                tf.keras.layers.LSTM(
                    nodes_layer,
                    return_sequences=False,
                    name=layer_name
                )
            )
        else:
            logger.info(f"Building sequential LSTM model - {layer_name}")
            model.add(
                tf.keras.layers.LSTM(
                    nodes_layer,
                    return_sequences=True,
                    name=layer_name
                )
            )

    # Dense Layers
    for layer_nr in range(1, nr_dense_layers + 1):
        layer_name = f"DENSE_{layer_nr}"
        nodes_layer = units_per_layer[layer_name]

        logger.info(f"Building sequential LSTM model - {layer_name}")

        model.add(
            tf.keras.layers.Dense(
                nodes_layer,
                activation=dense_layer_activation,
                name=layer_name
            )
        )

    # Last dense layer only one unit and no activation since it will predict the global temperature
    model.add(
        tf.keras.layers.Dense(1, name='DENSE_OUTPUT')
    )

    return model
