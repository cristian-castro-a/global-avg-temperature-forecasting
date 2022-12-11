import hydra
from omegaconf import DictConfig

from utils.handlers import HandlingData
from utils.printers import print_dataframe_descriptive_statistics
from utils.plotting import plot_lines_by


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Exploratory data analysis
    """

    # Load data: pip_dataset
    pip_dataset = HandlingData(csv_file=cfg.datafiles.pip_dataset, data_path=cfg.directories.data)
    pip_dataset_df = pip_dataset.load_raw_data()

    # Get the descriptive statistics: raw pip_dataset
    print_dataframe_descriptive_statistics(data=pip_dataset_df,
                                           file_name='raw_pip_dataset_df_descriptive_stats.csv',
                                           path_to_results=cfg.directories.eda_results_dir)

    # Let's make a plot for overall raw data
    plot_lines_by(data=pip_dataset_df,
                  plot_x='year',
                  plot_y='headcount_ratio_international_povline',
                  plot_by='country',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name='pip_dataset_df_povline.html')


if __name__ == '__main__':
    main()