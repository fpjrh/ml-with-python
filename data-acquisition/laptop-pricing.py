import pandas as pd
import numpy as np

file_path='/Users/fpj/Development/python/ml-with-python/data-acquisition/data/'
file_name = file_path + "laptop_pricing_dataset_base.csv"

df = pd.read_csv(file_name)

#filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
#df = pd.read_csv(filepath, header=None)

# Write your code below and press Shift+Enter to execute.
df = pd.read_csv(file_name, header=None)
#print(df.head())

headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

df.columns = headers
df.columns
#print(df.head(10))

df.replace('?',np.nan, inplace=True)
print(df.head(20))

# print the datatypes
print(df.dtypes)

# print the statistical summary
print(df.describe(include='all'))

# print the summary information
print(df.info())

