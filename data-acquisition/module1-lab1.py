# import pandas library
import pandas as pd
import numpy as np

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path='/Users/fpj/Development/python/ml-with-python/data-acquisition/data/'

#download(file_path, "auto.csv")
file_name = file_path + "auto.csv"

df = pd.read_csv(file_name)

#filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
#df = pd.read_csv(filepath, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head(5)

# show the last 10 rows using dataframe.tail() method
print("The last 10 rows of the dataframe\n")
df.tail(10)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

df.columns = headers
df.columns

df1=df.replace('?',np.NaN)

df=df1.dropna(subset=["price"], axis=0)
df.head(20)

# Write your code below and press Shift+Enter to execute 
print(df.columns)

df.to_csv("automobile.csv", index=False)

df.dtypes

# check the data type of data frame "df" by .dtypes
print(df.dtypes)

df.describe()