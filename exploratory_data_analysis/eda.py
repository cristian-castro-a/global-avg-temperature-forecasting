import hydra
from omegaconf import DictConfig
import logging
from utils.sdk_config import SDKConfig
from utils.parsers import read_data


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Exploratory data analysis
    """

    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)


if __name__ == '__main__':
    main()