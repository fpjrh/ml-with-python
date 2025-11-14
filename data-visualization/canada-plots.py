import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # visualization library


mpl.style.use('ggplot')  # optional: for ggplot-like style

# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) # >= 2.0.0

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
# check the shape
print(df_can.shape)

# Clean up some unnecessary columns
to_drop = ['AREA', 'REG', 'DEV', 'Type', 'Coverage']
df_can.drop(to_drop, inplace=True, axis=1)

# Rename some columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
# Add a 'Total' column that sums up the total
df_can['Total'] = df_can.sum(axis=1, numeric_only=True)

print("Cleaned up some unnecessary columns: \n", df_can.head(5))

# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace=True)
print("Set the Country column as the index: \n", df_can.head(5))

# create a list of years for which data is available
years = list(range(1980, 2014))

# Area plots
df_can.sort_values(by='Total', ascending=False, inplace=True)
# get the top 5 countries based on total immigration
df_top5 = df_can.head(5)
# transpose the data for plotting
df_top5 = df_top5[years].transpose()
print("Top 5 countries for immigration: \n", df_top5.head())
# change the index values of the DataFrame to integers for plotting
df_top5.index = df_top5.index.map(int)
df_top5.plot(kind='area', stacked=False, figsize=(20, 10))
plt.title('Immigration from Top 5 countries')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')
# show the plot
plt.show()
