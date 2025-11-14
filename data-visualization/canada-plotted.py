import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib as mpl
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

# Clean up some unnecessary columns
to_drop = ['AREA', 'REG', 'DEV', 'Type', 'Coverage']
df_can.drop(to_drop, inplace=True, axis=1)

# Rename some columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
# Add a 'Total' column that sums up the total
df_can['Total'] = df_can.sum(axis=1, numeric_only=True)

print("Cleaned up some unnecessary columns: \n", df_can.head(5))

# let's set the Country column as the index
df_can.set_index('Country', inplace=True)
print("Set the Country column as the index: \n", df_can.head(5))
#
years = list(range(1980, 2014))
# print the matplotlib version
print("Matplotlib version: ", mpl.__version__)
# apply a style
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style

# let's do a line plot on Haiti for 1980 to 2013
haiti = df_can.loc['Haiti', years]
print("Haiti df head:\n", haiti.head())
# plot the data
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to integer for plotting
haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.text(2000, 6000, '2010 Earthquake')
plt.show()

# compare the number of immigrants from India and China
df_CI = df_can.loc[['India', 'China'], years]
print("India and China df head:\n", df_CI.head())
df_CI = df_CI.transpose()
#
df_CI.index = df_CI.index.map(int) # change the index values of the DataFrame to integers for plotting
df_CI.plot(kind='line')
plt.title('Immigration from India and China')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.show()

# Plot the data for top 5 countries that immigrated to Canada
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
top5 = df_can.head(5)
print("Top 5 immigration countries: \n", top5)
top5 = top5[years].transpose()
top5.index = top5.index.map(int) # change the index values of the DataFrame to integers for plotting
top5.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size
plt.title('Immigration from Top 5 countries')
plt.xlabel('Year')
plt.ylabel('Number of immigrants')

plt.show()
