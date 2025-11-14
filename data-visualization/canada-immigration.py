import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib.pyplot as plt
import seaborn as sns  # visualization library

CANADA_IMMIGRATION_FILE = '/Users/fpj/Development/python/ml-with-python/data-visualization/data/canada-immigration.xlsx'

# Load the data
df_can = pd.read_excel(
    CANADA_IMMIGRATION_FILE,
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2
)
print("Data read into a pandas dataframe!")
# check the head and tail
print(df_can.head(5))
print(df_can.tail(5))
# get some basic info about the dataframe
print("Dataframe info: \n", df_can.info(verbose=False))
print("Dataframe columns: ", df_can.columns)
print("Dataframe shape", df_can.shape)
print("df_can.columns type: ", type(df_can.columns))
# turn the columns and index into lists
df_can.columns.tolist()
df_can.index.tolist()
print("df_can.columns type: ", type(df_can.columns.tolist()))
print("df_can.index type: ", type(df_can.index.tolist()))

# Clean up some unnecessary columns
to_drop = ['AREA', 'REG', 'DEV', 'Type', 'Coverage']
df_can.drop(to_drop, inplace=True, axis=1)
print("Cleaned up some unnecessary columns: \n", df_can.head(5))

# Rename some columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
# Add a 'Total' column that sums up the total
df_can['Total'] = df_can.sum(axis=1, numeric_only=True)

print("Columns renamed and total column added: \n", df_can.head(5))
# check how many null objects there are
# print("Number of null values: ", df_can.isnull().sum())

# let's see the countries
print("Countries in the dataframe: \n", df_can['Country'].unique())

# let's create a subset of the data by country for the years 1980 to 1985
df_1980s = df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]]
# print the first five rows
print("Subset of data for 1980s: \n", df_1980s.head())

# set the Country column as the index
df_can.set_index('Country', inplace=True)
# print the first five rows
print("Dataframe with index set to Country: \n", df_can.head())
# let's look at Japan
print("Japan entry: \n", df_can.loc['Japan'])
# let's look at Japan by index (87)
print("Japan entry by index: \n", df_can.iloc[87])
#
print("Japan entry by index: \n", df_can[df_can.index == 'Japan'])
# filter by criteria
condition = df_can['Continent'] == 'Asia'
print("Filtered by continent: \n", df_can[condition])

# filter by AreaName is Africa and Region is Southern Africa
condition = (df_can['Continent'] == 'Africa') & (df_can['Region'] == 'Southern Africa')
print("Filtered by AreaName and Region: \n", df_can[condition])

# sort the top5 immigration countries by descending order of immigration
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
top5 = df_can.head(5)
print("Top 5 immigration countries: \n", top5)

# find the top3 countries for the year 2010
df_can.sort_values(by=2010, ascending=False, axis=0, inplace=True)
top3 = df_can[2010].head(3)
print("Top 3 countries for 2010: \n", top3)
