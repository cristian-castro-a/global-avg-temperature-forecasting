import pandas as pd
from exploratory_data_analysis import config
from utils.plotting import plot_lines_by
import matplotlib.pyplot as plt
import seaborn as sns


# OIL Consumption Dataset
df_oil_consum_pc = pd.read_csv(config.oil_consum_pc)
df_oil_consum_pc.head()
df_oil_consum_pc.info()

df_oil_consum_pc.drop('Code', axis=1, inplace=True)

df_oil_consum_pc.info()

# group by year
df_oil_consum_pc = df_oil_consum_pc.groupby(by=['Year']).sum().reset_index()[['Year','Oil per capita (kWh)']]
df_oil_consum_pc.insert(loc = 1, column = 'month', value = 12)
df_oil_consum_pc.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_oil_consum_pc[['Year','month','day']])
df_oil_consum_pc.insert(loc = 0, column = 'date', value = values)
df_oil_consum_pc.drop(['Year','month','day'], axis = 1, inplace = True)
df_oil_consum_pc.set_index('date', inplace = True)
df_oil_consum_pc.head()
df_oil_consum_pc.rename(columns={'Oil per capita (kWh)': 'oil_consum_kWh'}, inplace=True)


# visualizing
plt.figure(figsize=(22,10))
plt.plot(df_oil_consum_pc.iloc[1:])
plt.title("Total world oil consumption", fontsize=20)
plt.xlabel("Date", fontsize=15)
plt.ylabel("Oil in kWh", fontsize=15)
plt.show()


# GLOBAL TEMP
df_global_temp = pd.read_csv(config.global_temp, sep="     ", header=None, names=["year", "temp_no_smoothing", "temp_lowess_5"])
df_global_temp.head()
df_global_temp.info()

df_global_temp.drop(index=[0,1,2,3], inplace=True)
df_global_temp.reset_index(drop=True, inplace=True)
df_global_temp.info()

# year -> to_datetime
df_global_temp.insert(loc = 1, column = 'month', value = 12)
df_global_temp.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_global_temp[['year','month','day']])
df_global_temp.insert(loc = 0, column = 'date', value = values)
df_global_temp.drop(['year','month','day'], axis = 1, inplace = True)
df_global_temp.set_index('date', inplace = True)
df_global_temp.head()

df_global_temp.rename(columns={'Oil per capita (kWh)': 'oil_consum_kWh'}, inplace=True)

# join df_global_temp and df_oil_consum_pc (left join)
df_final = df_global_temp.join(df_oil_consum_pc)


# read ocean warming data
df_owid_co2 = pd.read_csv(config.owid_co2)
df_owid_co2.info()

df_owid_co2 = df_owid_co2.groupby(by=['year']).sum().reset_index()[['year','co2', 'population','gdp']]
df_owid_co2.insert(loc = 1, column = 'month', value = 12)
df_owid_co2.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_owid_co2[['year','month','day']])
df_owid_co2.insert(loc = 0, column = 'date', value = values)
df_owid_co2.drop(['year','month','day'], axis = 1, inplace = True)
df_owid_co2.set_index('date', inplace = True)
df_owid_co2.head()

# join df_final and df_owid_co2 (left join)
df_final = df_final.join(df_owid_co2)


# Ocean Warming data - not working
#df_ocean_warming = pd.read_json(config.ocean_warming, typ='series')
#df_ocean_warming.info()
#df_ocean_warming=df_ocean_warming.to_frame()
#df_ocean_warming.head()
#df_final = df_final.join(df_ocean_warming)




# renewable_share_energy
df_renewable_share_energy = pd.read_csv(config.renewable_share_energy)
df_renewable_share_energy.info()
df_renewable_share_energy.rename(columns={'Renewables (% equivalent primary energy)': 'renewables'}, inplace=True)

df_renewable_share_energy = df_renewable_share_energy.groupby(by=['Year']).sum().reset_index()[['Year','renewables']]
df_renewable_share_energy.insert(loc = 1, column = 'month', value = 12)
df_renewable_share_energy.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_renewable_share_energy[['Year','month','day']])
df_renewable_share_energy.insert(loc = 0, column = 'date', value = values)
df_renewable_share_energy.drop(['Year','month','day'], axis = 1, inplace = True)
df_renewable_share_energy.set_index('date', inplace = True)


df_final = df_final.join(df_renewable_share_energy)



# Electricity Production
df_elec_prod = pd.read_csv(config.elec_prod)
df_elec_prod.info()

df_elec_prod.groupby('Unit').groups.keys() #only GWh -> sum up while grouping

df_elec_prod['Time'] = df_elec_prod.Time.str.extract('(\d+)')
df_elec_prod = df_elec_prod.groupby(by=['Time']).sum().reset_index()[['Time','Value']]
# leave electricity production out, as it contains only last 12 years


# World employment rate
df_employ_rate = pd.read_csv(config.employ_rate)
df_employ_rate.info()
df_employ_rate = df_employ_rate.groupby(by=['TIME']).mean().reset_index()[['TIME','Value']]
df_employ_rate.rename(columns={'TIME': 'year', 'Value':'employment_rate'}, inplace=True)


df_employ_rate.insert(loc = 1, column = 'month', value = 12)
df_employ_rate.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_employ_rate[['year','month','day']])
df_employ_rate.insert(loc = 0, column = 'date', value = values)
df_employ_rate.drop(['year','month','day'], axis = 1, inplace = True)
df_employ_rate.set_index('date', inplace = True)

df_final = df_final.join(df_employ_rate)


# global energy substitutaion
df_global_energy = pd.read_csv(config.global_energy)
df_global_energy.info()

df_global_energy.drop(index=[0,1,2,3,4,5,6,7,8,9,19,10,11,12,13,14,15,16], inplace=True)
df_global_energy.reset_index(drop=True, inplace=True)
df_global_energy.drop(['Entity', 'Code'], axis = 1, inplace = True)

df_global_energy.rename(columns={'Other renewables (TWh, substituted energy)': 'other_renewables',
                                 'Biofuels (TWh, substituted energy)':'biofuels',
                                 'Solar (TWh, substituted energy)':'solar',
                                 'Wind (TWh, substituted energy)':'wind',
                                 'Hydropower (TWh, substituted energy)':'hydropower',
                                 'Nuclear (TWh, substituted energy)':'nuclear',
                                 'Gas (TWh, substituted energy)':'gas',
                                 'Oil (TWh, substituted energy)':'oil',
                                 'Coal (TWh, substituted energy)':'coal',
                                 'Traditional biomass (TWh, substituted energy)':'trad_biomass'}, inplace=True)

df_global_energy.insert(loc = 1, column = 'month', value = 12)
df_global_energy.insert(loc = 2, column = 'day', value = 31)
values = pd.to_datetime(df_global_energy[['Year','month','day']])
df_global_energy.insert(loc = 0, column = 'date', value = values)
df_global_energy.drop(['Year','month','day'], axis = 1, inplace = True)
df_global_energy.set_index('date', inplace = True)

df_final = df_final.join(df_global_energy)

# data cleaning
df_final.info() # 142 rows in total if no NaN

df_final.isnull().sum()
# df_final.isnull().sum()
# temp_no_smoothing     0
# temp_lowess_5         0
# oil_consum_kWh       85 -> more than 50% missing - delete or impute? correlation with temp between 0.75 and 0.77
# co2                   1
# population            1
# gdp                   1
# renewables           85 -> more than 50% missing - delete or impute? correlation with temp between 0.93 and 0.97
# employment_rate      75 -> more than 50% missing - delete as correlation only 0.54
# other_renewables     86
# biofuels             86
# solar                86
# wind                 86
# hydropower           86
# nuclear              86
# gas                  86
# oil                  86
# coal                 86
# trad_biomass         86


correlation_mat = df_final.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()


# drop employment rate, solar and trad_biomass column due to missing values and low correlation
df_final.drop(['employment_rate', 'solar', 'trad_biomass'], axis = 1, inplace = True)

# drop until year 1965
df_final.drop(index=df_final.index[:85], axis=0, inplace=True)

#impute missing values
df_final.isnull().sum()

# visualize features
plt.figure(figsize=(16, 8), dpi=150)
df_final['co2'].plot(label='co2', color='orange')
df_final['population'].plot(label='population')
df_final['gdp'].plot(label='gdp')

# adding title to the plot
plt.title('Development of features')

# adding Label to the x-axis
plt.xlabel('Years')

# adding legend to the curve
plt.legend()




df_final.to_csv(r'~/Desktop/RWTH_DDS/EMECS/Project_TimeSeries/poverty-forecasting/data/yearly_aggr_data.csv')




