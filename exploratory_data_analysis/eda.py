import hydra
from omegaconf import DictConfig

from utils.parsers import CountryPoverty
from utils.writers import print_dataframe_descriptive_statistics
from utils.plotting import plot_lines_by


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Exploratory data analysis
    """

    # Load data: pip_dataset for Chile
    Chile_2017 = CountryPoverty(csv_file=cfg.datafiles.pip_dataset,
                                data_path=cfg.directories.data,
                                ppp_version=cfg.data_filters.ppp_version,
                                country_name='Chile').load_country_data()

    # # Get the descriptive statistics
    print_dataframe_descriptive_statistics(data=Chile_2017,
                                           file_name='raw_chile_2017_descriptive_stats.csv',
                                           path_to_results=cfg.directories.eda_results_dir)

    # Let's make a plot for Chile raw data
    plot_lines_by(data=Chile_2017,
                  plot_x='year',
                  plot_y='headcount_ratio_international_povline',
                  plot_by='country',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name='Chile_2017_povline.html')

    # Load data: pip_dataset for India
    India_2017 = CountryPoverty(csv_file=cfg.datafiles.pip_dataset,
                                data_path=cfg.directories.data,
                                ppp_version=cfg.data_filters.ppp_version,
                                country_name='India').load_country_data()

    # # Get the descriptive statistics
    print_dataframe_descriptive_statistics(data=India_2017,
                                           file_name='raw_india_2017_descriptive_stats.csv',
                                           path_to_results=cfg.directories.eda_results_dir)

    # Let's make a plot for Chile raw data
    plot_lines_by(data=India_2017,
                  plot_x='year',
                  plot_y='headcount_ratio_international_povline',
                  plot_by='country',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name='India_2017_povline.html')


if __name__ == '__main__':
    main()