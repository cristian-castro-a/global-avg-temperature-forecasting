import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils.exploratory_data_analysis import get_eda_summary, plot_time_series
from utils.parsers import read_data
from utils.preprocessing import preprocess_data
from utils.sdk_config import SDKConfig


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:
    """
    Exploratory data analysis
    """
    '''
    To Do:
    1. Access config
    2. Create nested loops for all combinations of p,d,q
    3. Save Model output to txt
    4. Forecast using all the arima models
    5. Save all forecasts
    6. Compare MAPE of all the forecasts and find the best one
    7. Save the best model
    '''
    logger.info('Setting working directories for the project')
    SDKConfig().set_working_dirs()

    data_dict = read_data(data_dir=SDKConfig().data_dir, config=config)
    processed_data_dict = preprocess_data(data_dict=data_dict)

    # Visualization of predictors and summary of statistics
    path_to_results = SDKConfig().get_output_dir("plots_predictors")
    plot_time_series(processed_data_dict=processed_data_dict, path_to_results=path_to_results)
    eda_summary = get_eda_summary(processed_data_dict=processed_data_dict)


if __name__ == '__main__':
    main()