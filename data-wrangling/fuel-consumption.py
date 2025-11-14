import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# constants
FILE_PATH = '/Users/fpj/Development/python/ml-with-python/data-acquisition/data/'
car_file = FILE_PATH + "auto.csv"

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# open up the file with the headers
df = pd.read_csv(car_file, names=headers)

#let's take a peek
#print(df.head(5))

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
print(df.head(5))

# see if missing data is null
missing_data = df.isnull()
missing_data.head(5)
# count the missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

# start with replacement for simple imputation
# replace the missing values with the average of the column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df.replace({"normalized-losses": {np.nan: avg_norm_loss}}, inplace=True)
#df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
print(df.head(5))

# replace the bore column missing values with the average of the column
avg_bore = df['bore'].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)
df.replace({"bore": {np.nan: avg_bore}}, inplace=True)
#df["bore"].replace(np.nan, avg_bore, inplace=True)
print(df.head(5))

# replace the stroke column missing values with the average of the column
avg_stroke = df['stroke'].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df.replace({"stroke": {np.nan: avg_stroke}}, inplace=True)
#df["stroke"].replace(np.nan, avg_stroke, inplace=True)
print(df.head(5))


# the doors
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 
df.replace({"num-of-doors":{np.nan, "four"}}, inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)

# check your transformed data 
print(df.head())


# let's do some binning of horsepower

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df.replace({"horsepower": {np.nan: avg_horsepower}}, inplace=True)

df["horsepower"]=df["horsepower"].astype(int, copy=True)
plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()
# bin in 3 bins
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
# set group names
group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

plt.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

