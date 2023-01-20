import logging

import hydra
from omegaconf import DictConfig

from utils.exploratory_data_analysis import get_eda_summary
from utils.parsers import read_data
from utils.plotting import plot_lines_by
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

    test_dict = {k: processed_data_dict[k] for k in ('co2_emissions', 'global_temperature')}
    eda_summary = get_eda_summary(processed_data_dict=test_dict)


if __name__ == '__main__':
    main()