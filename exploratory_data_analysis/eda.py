import logging

import hydra
from omegaconf import DictConfig

from utils.parsers import read_data
from utils.preprocessing import preprocess_data
from utils.sdk_config import SDKConfig


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Exploratory data analysis
    """
    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)
    processed_data_dict = preprocess_data(data_dict=data_dict)


if __name__ == '__main__':
    main()