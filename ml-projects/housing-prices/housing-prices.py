# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# constants
FILEPATH = '/Users/fpj/Development/python/ml-with-python/ml-projects/housing-prices/data/kc_house_data.csv'

# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv(FILEPATH)
print(df.head())
# Step 2: Data Wrangling
# display the data types of each column
print(df.dtypes)
print(df.describe())
# drop the columns id and unnamed: 0
df = df.drop(['id', 'Unnamed: 0'], axis=1)
# display the first few rows of the dataframe
print(df.describe())
# see if we have missing values for bedrooms and bathrooms
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
# replace the missing values with the mean of the column
df['bedrooms'] = df['bedrooms'].replace(np.nan, df['bedrooms'].mean())
df['bathrooms'] = df['bathrooms'].replace(np.nan, df['bathrooms'].mean())
# see if we have missing values for bedrooms and bathrooms
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
# Step 3: Exploratory Data Analysis

# count the number of houses with unique floor values and convert the result to a dataframe
print(pd.DataFrame(df['floors'].value_counts()))
# use the boxplot function to determine whether houses with a waterfront view or without a waterfront view have more price outliers
sns.boxplot(x='waterfront', y='price', data=df)
plt.show()
# use the regplot function to determine if the feature sqft_above is negatively or positively correlated with price
sns.regplot(x='sqft_above', y='price', data=df)
plt.show()
# use the correlation heatmap function to determine which features are correlated to each other
# convert the date column to a datetime object
df['date'] = pd.to_datetime(df['date'])
#
corr = df.corr()
print(corr['price'].sort_values(ascending=True))
# Step 4: Model Development
# Fit a linear regression model using the longitude feature 'long'
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)
print("\nScore for feature 'long'", lm.score(X, Y))
# fir a linear regression model to predict price using the feature 'sqft_living' then calculate the R^2
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print("R2", metrics.r2_score(Y, lm.predict(X)))
lm.score(X, Y)
print("Score for feature 'sqft_living'", lm.score(X, Y))
#
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
# fir a linear regression model to predict price using all the features then calculate the R^2
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print("R2", metrics.r2_score(Y, lm.predict(X)))
lm.score(X, Y)
print("Score for all features", lm.score(X, Y))
# create a list of tuples the first element contains the estimator and the second element contains the model constructor
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
# use the list Input to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2
pipe=Pipeline(Input)
pipe.fit(X,Y)
print("R2", metrics.r2_score(Y, pipe.predict(X)))

# Step 5: Model Evaluation and Refinement
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("\nnumber of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
# create and fit a Ridge regression model, alpha=0.1, and calculate the R^2 using the test data
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(x_train, y_train)
print("Ridge R2", metrics.r2_score(y_test, ridge_reg.predict(x_test)))
# perform a second order polynomial transform on both the training data and test data; create and fit a Ridge regression model, alpha=0.1, and calculate the R^2 using the test data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
polynomial_features= PolynomialFeatures(degree=2)
ridge_poly = make_pipeline(polynomial_features, ridge_reg)
ridge_poly.fit(x_train, y_train)
print("Ridge Poly R2", metrics.r2_score(y_test, ridge_poly.predict(x_test)))
