import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import folium
#from js import fetch
import io

FILE_PATH = "/Users/fpj/Development/Python/ml-with-python/data-visualization/data/"
FILE_NAME = "Historical_Wildfires.csv"

# Read the data
df = pd.read_csv(FILE_PATH + FILE_NAME)
print('Data read into a pandas dataframe!')

# Display the first few rows of the dataframe
print(df.head())
# verify column names
print(df.columns)
# check the data types
print(df.dtypes)
# fix the date column
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
#
print(df.columns)
print(df.dtypes)
# Plot the change in average estimaed fire area over time
plt.figure(figsize=(12  , 6))
# Group by year and calculate the mean of Estimated_fire_area
df_new = df.groupby('Year')['Estimated_fire_area'].mean()
# Plot the data using pandas
df_new.plot(x=df_new.index, y=df_new.values, kind='line')
plt.xlabel('Year')
plt.ylabel('Average Estimated Fire Area (km2)')
plt.title('Change in Average Estimated Fire Area Over Time')
plt.show()
# Plot the change from 2010 to 2013 by month
plt.figure(figsize=(12, 6))
# Filter the data for years 2010 and 2013
df_narrowed = df[(df['Year'] >= 2010) & (df['Year'] <= 2013)]
# Group by month and calculate the mean of Estimated_fire_area
df_monthly = df_narrowed.groupby('Month')['Estimated_fire_area'].mean()
# Plot the data using pandas
df_monthly.plot(x=df_monthly.index, y=df_monthly.values, kind='line')
plt.xlabel('Month')
plt.ylabel('Average Estimated Fire Area (km2)')
plt.title('Change in Average Estimated Fire Area from 2010 to 2013')
plt.show()

# Identify the regions
print(df['Region'].unique())
# Plot the distribution of mean estimated fire brightness by region using seaborn barplot
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Mean_estimated_fire_brightness', data=df)
plt.xlabel('Region')
plt.ylabel('Mean Estimated Fire Brightness')
plt.title('Distribution of Mean Estimated Fire Brightness by Region (Kelvin)')
plt.xticks(rotation=45)
plt.show()

# Group the data by region and find the sum of count of fires
df_region = df.groupby('Region')['Count'].sum().reset_index()
region_counts = df.groupby('Region')['Count'].sum()
print(df_region.head())
# Plot the portion of count of pixels for presumed vegetation fires by region using pie chart
plt.figure(figsize=(12, 6))
plt.pie(df_region['Count'], labels=region_counts.index )
plt.axis('equal')
plt.title('Portion of Count of Pixels for Presumed Vegetation Fires by Region')
plt.legend([(i,round(k/region_counts.sum()*100,2)) for i,k in zip(region_counts.index, region_counts)])
plt.show()

# Creating a histogram to visualize the distribution of mean estimated fire brightness
plt.figure(figsize=(10, 6))
# Using plt.hist to create the histogram
# Setting the number of bins to 20 for better visualization
plt.hist(x=df['Mean_estimated_fire_brightness'], bins=20)
plt.xlabel('Mean Estimated Fire Brightness (Kelvin)')
plt.ylabel('Count')
plt.title('Histogram of Mean Estimated Fire Brightness')
plt.show()

# Creating a histogram to visualize the distribution of mean estimated fire brightness across regions using Seaborn
# Using sns.histplot to create the histogram
# Specifying the DataFrame (data=df) and the column for the x-axis (x='Mean_estimated_fire_brightness')
# Adding hue='Region' to differentiate the distribution across regions
sns.histplot(data=df, x='Mean_estimated_fire_brightness', hue='Region', multiple='stack')
plt.show()

# Creating a scatter plot to visualize the relationship between mean estimated fire radiative power and mean  confidence using Seaborn
plt.figure(figsize=(8, 6))
# Using sns.scatterplot to create the scatter plot
# Specifying the DataFrame (data=df) and the columns for the x-axis (x='Mean_confidence') and y-axis            (y='Mean_estimated_fire_radiative_power')
sns.scatterplot(data=df, x='Mean_confidence', y='Mean_estimated_fire_radiative_power')
plt.xlabel('Mean Estimated Fire Radiative Power (MW)')
plt.ylabel('Mean Confidence')
plt.title('Mean Estimated Fire Radiative Power vs. Mean Confidence')
plt.show()

region_data = {'region':['NSW','QL','SA','TA','VI','WA','NT'], 'Lat':[-31.8759835,-22.1646782,-30.5343665,-42.035067,-36.5986096,-25.2303005,-19.491411], 
               'Lon':[147.2869493,144.5844903,135.6301212,146.6366887,144.6780052,121.0187246,132.550964]}
reg=pd.DataFrame(region_data)
print(reg)
# instantiate a feature group 
aus_reg = folium.map.FeatureGroup()

# Create a Folium map centered on Australia
Aus_map = folium.Map(location=[-25, 135], zoom_start=4)

# loop through the region and add to feature group
for lat, lng, lab in zip(reg.Lat, reg.Lon, reg.region):
    aus_reg.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            popup=lab,
            radius=5, # define how big you want the circle markers to be
            color='red',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
Aus_map.add_child(aus_reg)
Aus_map.save('aus_map.html')
print('Map saved!')
# Display the map
Aus_map._build_map()
mapWidth, mapHeight = (400,500) # width and height of the displayed iFrame, in pixels
srcdoc = Aus_map.HTML.replace('"', '&quot;')
embed = HTML('<iframe srcdoc="{}" '
             'style="width: {}px; height: {}px; display:block; width: 50%; margin: 0 auto; '
             'border: none"></iframe>'.format(srcdoc, width, height))
embed
#display(Aus_map)
#print('Map displayed!')
