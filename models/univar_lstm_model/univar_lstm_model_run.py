import logging

import hydra
from omegaconf import DictConfig

from build_uni_lstm import univariate_lstm
from utils.parsers import read_data
from utils.preprocessing import preprocess_data, Windowing
from utils.sdk_config import SDKConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Univariate LSTM Model
    """
    logger.info('Setting working directories for the project')
    SDKConfig().set_working_dirs()

    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)
    processed_data_dict = preprocess_data(data_dict=data_dict)

    sequences_generator = Windowing(df=processed_data_dict['global_temperature'], mode='Univariate')
    X, y = sequences_generator.get_input_sequences(on_column='global_temperature')

    model = univariate_lstm(X_train=X, config=config)

    print('test')


if __name__ == '__main__':
    main()