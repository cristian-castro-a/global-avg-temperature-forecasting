import hydra
import pandas as pd
from omegaconf import DictConfig
from exploratory_data_analysis import config
from utils.parsers import CSVParser
from utils.writers import print_dataframe_descriptive_statistics
from utils.plotting import plot_lines_by


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Exploratory data analysis
    """
    df_world_population = CSVParser(csv_file= cfg.datafiles.world_population, data_path=cfg.directories.data).load_raw_data()
    ## This dataframe contains population from year 1950 to forecasted year 2100
    # df_world_population.info()

    df_world_employment = CSVParser(csv_file=cfg.datafiles.world_employment_rate , data_path= cfg.directories.data).load_raw_data()
    ## This dataframe contains employment rate based for 44 location from year 1979 to 2021
    # print(df_world_employment.nunique())
    # df_world_employment.info()

    df_world_electricity = CSVParser(csv_file=cfg.datafiles.world_electricity_production ,data_path= cfg.directories.data).load_raw_data()
    ## This dataframe contains monthly electricity consumption data from 2010 to 2022 for 53 countries
    # print(df_world_electricity.nunique())

    df_renewable_energy = CSVParser(csv_file=cfg.datafiles.renewable_share_energy, data_path = cfg.directories.data).load_raw_data()
    ## This dataframe contains percentage of renewable energy used by 114 entities from 1965 to 2021
    # print(df_renewable_energy.head())
    # print(df_renewable_energy.tail())
    # print(df_renewable_energy.nunique())

    df_co2_emission = CSVParser(csv_file=cfg.datafiles.world_co2, data_path= cfg.directories.data).load_raw_data()
    df_co2_emission.info()
    print(df_co2_emission.nunique())

    # df_ocean_warming = pd.read_json(cfg.datafiles.ocean_warming)

    plot_lines_by(data=df_world_population,
                  plot_x='date',
                  plot_y=' Population',
                  plot_by= 'date',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name=f"{cfg.data_filters.ppp_version}_population.html")

    plot_lines_by(data=df_world_employment,
                  plot_x='TIME',
                  plot_y='Value',
                  plot_by='LOCATION',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name=f"{cfg.data_filters.ppp_version}_employment.html")

    plot_lines_by(data=df_world_electricity,
                  plot_x='Time',
                  plot_y='Value',
                  plot_by='Country',
                  path_to_results=cfg.directories.eda_results_dir,
                  file_name=f"{cfg.data_filters.ppp_version}_electricity.html")

    plot_lines_by(data=df_renewable_energy,
                    plot_x='Year',
                    plot_y='Renewables (% equivalent primary energy)',
                    plot_by='Entity',
                    path_to_results=cfg.directories.eda_results_dir,
                    file_name=f"{cfg.data_filters.ppp_version}_renewable.html")

    plot_lines_by(data=df_co2_emission,
                    plot_x='year',
                    plot_y='co2',
                    plot_by='country',
                    path_to_results=cfg.directories.eda_results_dir,
                    file_name=f"{cfg.data_filters.ppp_version}_CO2_emission.html")

    plot_lines_by(data=df_co2_emission,
                    plot_x='year',
                    plot_y='primary_energy_consumption',
                    plot_by='country',
                    path_to_results=cfg.directories.eda_results_dir,
                    file_name=f"{cfg.data_filters.ppp_version}_energy_consumption.html")
if __name__ == '__main__':
    main()