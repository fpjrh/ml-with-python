"""
Objectives:

After completing this lab you will be able to:

- Create informative and visually appealing plots with Matplotlib and Seaborn.
- Apply visualization to communicate insights from the data.
- Analyze data through using visualizations.
- Customize visualizations

"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium
#
FILE_PATH = '/Users/fpj/Development/python/ml-with-python/data-visualization/final-project/data/'
FILE_NAME = 'historical_automobile_sales.csv'
STATES_JSON = 'us-states.json'

# Read the data from the CSV file
df = pd.read_csv(FILE_PATH + FILE_NAME)
# Display the first few rows of the DataFrame
print(df.head())
print(df.columns)
# Create a line chart using pandas to show automobile sales over the years
# Group the data by year and calculate average Automobile Sales
avg_sales_by_year = df.groupby('Year')['Automobile_Sales'].mean()
# Create the line chart
#avg_sales_by_year.plot(x='Year', y='Automobile_Sales', kind='line', marker='o')
#plt.xlabel('Year')
#plt.ylabel('Average Automobile Sales')
#plt.title('Average Automobile Sales by Year')
#plt.grid(True)
#plt.show()
# Do the same plot identiyfing years of recession

plt.figure(figsize=(10, 6))
avg_sales_by_year.plot(x='Year', y='Automobile_Sales', kind='line', marker='o', color='green')
plt.xticks(list(range(1980,2024)), rotation = 75)
plt.xlabel('Year')
plt.ylabel('Average Automobile Sales')
plt.title('Average Automobile Sales by Year')
plt.grid(False)
# Add ticks for recession years
#plt.xticks(recession_years, rotation=45)
plt.text(1983, 650, '1980-82 Recession')
plt.text(1992, 650, '1991 Recession')
plt.text(2002, 650, '2000-01 Recession')
plt.text(2010, 650, '2007-09 Recession')
plt.legend()
plt.show()

# Copy the data frame to a new data frame
df_recession = df.copy()
recession_years = [1980, 1981, 1982, 1991, 2000, 2001, 2007, 2008,  2009, 2020]
# Add a column to identify recession years
#df_recession['Recession'] = df_recession['Year'].apply(lambda x: 1 if x in recession_years else 0)
#print('df_recession head : ', df_recession.head())
#print('df_recession columns : ', df_recession.columns)

# Create a plot with individual lines for each vehicle type of average vehicle sales for recession years only
# Group the data by year and vehicle type for recession years only
#df_recession_grouped = df_recession[df_recession['Recession'] == 1]
df_recession_grouped = df_recession.copy()
#
df_Mline = df_recession_grouped[['Vehicle_Type', 'Year', 'Automobile_Sales']]
df_Mline = df_Mline.groupby(['Year', 'Vehicle_Type'])['Automobile_Sales'].sum().reset_index()
#print(df_Mline.head())
fig, ax = plt.subplots()
df_Mline.groupby('Vehicle_Type').plot(x='Year', y='Automobile_Sales', ax=ax, kind='line', marker='o')
plt.xlabel('Year')
plt.ylabel('Average Automobile Sales')
plt.title('Average Automobile Sales by Year for Recession Years')
plt.grid(True)
plt.text(1983, 250, '1980-82 Recession')
plt.text(1992, 250, '1991 Recession')
plt.text(2002, 250, '2000-01 Recession')
plt.text(2010, 250, '2007-09 Recession')
plt.legend(df_Mline['Vehicle_Type'].unique())
plt.show()

#
new_df = df_recession.groupby('Recession')['Automobile_Sales'].mean().reset_index()
# Create a Seaborn barplot to compare average automobile sales for recession and non-recession years
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession', data=new_df, ax=ax)
plt.xlabel('Recession')
plt.ylabel('Average Automobile Sales')
plt.title('Average Automobile Sales by Year for Recession and Non-Recession Years')
plt.xticks([0, 1], ['Non-Recession', 'Recession'])
plt.show()

# do above for different vehicle types
recession_data = df[df['Recession'] == 1]
non_recession_data = df[df['Recession'] == 0]
new_df = df.groupby(['Recession', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
sales_by_vehicle_type_rec = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
sales_by_vehicle_type_nonrec = non_recession_data.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
print(new_df.head(20))
# Create a Seaborn grouped barchart to compare average automobile sales for recession and non-recession years
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Vehicle_Type', y='Automobile_Sales', hue='Recession', data=new_df, ax=ax)
plt.ylabel('Average Automobile Sales')
plt.title('Average Automobile Sales by Year for Recession and Non-Recession Years')
plt.legend(['Non-Recession', 'Recession'])
plt.show()

#
#
#
#Create dataframes for recession and non-recession period
rec_data = df[df['Recession'] == 1]
non_rec_data = df[df['Recession'] == 0]

#Figure
fig=plt.figure(figsize=(12, 6))

#Create different axes for subploting
ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2 ) # add subplot 2 (1 row, 2 columns, second plot). 

#plt.subplot(1, 2, 1)
sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0)
ax0.set_xlabel('Year')
ax0.set_ylabel('GDP')
ax0.set_title('GDP Variation during Recession Period')

#plt.subplot(1, 2, 2)
sns.lineplot(x='Year', y='GDP', data=non_rec_data, label='Non-Recession', ax=ax1)
ax1.set_xlabel('Year')
ax1.set_ylabel('GDP')
ax1.set_title('GDP Variation during non-Recession Period')

plt.tight_layout()
plt.show()

#------------------------------------------------Alternatively--------------
#Using subplot()
#plt.figure(figsize=(............, ..........))

#subplot 1
#plt.subplot(1, 2, 1)
#sns.lineplot(x='.........', y='......', data=.........., label='......')
#plt.xlabel('.......')
#plt.ylabel('..........')
#plt.legend()
#subplot 1
#plt.subplot(1, 2, 2)
#sns.lineplot(x='.........', y='......', data=.........., label='......')
#plt.xlabel('.......')
#plt.ylabel('..........')
#plt.legend()

#plt.tight_layout()
#plt.show()

#
# Create a bubble plot to visualize impact of seasonality on automobile sales
non_rec_data = df[df['Recession'] == 0]
    
size=non_rec_data['Seasonality_Weight'] #for bubble effect

sns.scatterplot(data=non_rec_data, x='Month', y='Automobile_Sales', size=size, hue='Seasonality_Weight', legend=False)

#you can further include hue='Seasonality_Weight', legend=False)

plt.xlabel('Month')
plt.ylabel('Automobile_Sales')
plt.title('Seasonality impact on Automobile Sales')

plt.show()

#
# Use matplotlib to create a scatter plot to visualize the relationship between average vehicle price to sales volume during recession
plt.figure(figsize=(10, 6))
#Create dataframes for recession and non-recession period
rec_data = df[df['Recession'] == 1]
plt.scatter(rec_data['Price'], rec_data['Automobile_Sales'])

plt.xlabel('Avg Vehicle Price')
plt.ylabel('Sales Volume')
plt.title('Relationship between Average Vehicle Price and Sales Volume during Recession')
plt.show()

# Filter the data 
Rdata = df[df['Recession'] == 1]
NRdata = df[df['Recession'] == 0]

# Calculate the total advertising expenditure for both periods
RAtotal = Rdata['Advertising_Expenditure'].sum()
NRAtotal = NRdata['Advertising_Expenditure'].sum()

# Create a pie chart for the advertising expenditure 
plt.figure(figsize=(8, 6))

labels = ['Recession', 'Non-Recession']
sizes = [RAtotal, NRAtotal]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title('Advertising Expenditure during Recession and Non-Recession Periods')

plt.show()
#
# Create a pie chart to display the total Advertising expenditure for each vehicle type during recession
# Filter the data 
Rdata = df[df['Recession'] == 1]

# Calculate the sales volume by vehicle type during recessions
VTexpenditure = Rdata.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()

# Create a pie chart for the share of each vehicle type in total expenditure during recessions
plt.figure(figsize=(8, 6))

labels = VTexpenditure.index
sizes = VTexpenditure.values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title('Advertising Expenditure by Vehicle Type during Recession')

plt.show()

#
# Create a lineplot to analyse the effect of unemployment rate on vehicle type and sales during recession
df_rec = df[df['Recession']==1]
sns.lineplot(data=df_rec, x='unemployment_rate', y='Automobile_Sales',
             hue='Vehicle_Type', style='Vehicle_Type', markers='o', err_style=None)
plt.ylim(0,850)
plt.legend(loc=(0.05,.3))
plt.show()
#
# Filter the data for the recession period and specific cities
recession_data = df[df['Recession'] == 1]

# Calculate the total sales by city
sales_by_city = recession_data.groupby('City')['Automobile_Sales'].sum().reset_index()

# Create a base map centered on the United States
map1 = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

# Create a choropleth layer using Folium
choropleth = folium.Choropleth(
    geo_data= 'us-states.json',  # GeoJSON file with state boundaries
    data=sales_by_city,
    columns=['City', 'Automobile_Sales'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Automobile Sales during Recession'
).add_to(map1)


# Add tooltips to the choropleth layer
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'], labels=True)
)

# Display the map
map1.save('recession_map.html')