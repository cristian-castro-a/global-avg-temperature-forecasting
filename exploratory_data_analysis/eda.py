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
    data_dict = {}
    for country in ['Chile', 'India', 'Germany']:
        data_dict[country] = CountryPoverty(csv_file=cfg.datafiles.pip_dataset,
                                            data_path=cfg.directories.data,
                                            ppp_version=cfg.data_filters.ppp_version,
                                            country_name=country,
                                            reporting_level='national').load_country_data()

    for name, df in data_dict.items():
        print_dataframe_descriptive_statistics(data=df,
                                               file_name=f"raw_{name}_{cfg.data_filters.ppp_version}_stats.csv",
                                               path_to_results=cfg.directories.eda_results_dir)
        plot_lines_by(data=df,
                      plot_x='year',
                      plot_y='headcount_ratio_international_povline',
                      plot_by='country',
                      path_to_results=cfg.directories.eda_results_dir,
                      file_name=f"{name}_{cfg.data_filters.ppp_version}_povline.html")


if __name__ == '__main__':
    main()