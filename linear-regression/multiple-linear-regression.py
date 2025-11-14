import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# Load the data
FILE_NAME = '/Users/fpj/Development/python/ml-with-python/exploratory-data-analysis/data/automobileEDA.csv'

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

df = pd.read_csv(FILE_NAME, header=0)
print(df.head())

print("\n\nModel 1: Using highway-mpg as predictor")

lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
# let's look at predictions
Yhat = lm.predict(X)
print(Yhat[0:5])
print("The intercept is ", lm.intercept_)
print("The slope is ", lm.coef_)

print("\n\nModel 2: Using engine-size as predictor")
# second lm
lm1 = LinearRegression()
X = df[['engine-size']]
Y = df['price']
lm1.fit(X, Y)
Yhat = lm1.predict(X)
print(Yhat[0:5])
print("The intercept is ", lm1.intercept_)
print("The slope is ", lm1.coef_)

print("\n\nModel 3: Using horsepower, curb-weight, engine-size, highway-mpg as predictors")
# MLR
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
Yhat = lm.predict(Z)
print(Yhat[0:5])
print("The intercept is ", lm.intercept_)
print("The slope is ", lm.coef_)

print("\n\nModel 4: Using normalized-losses, highway-mpg as predictors")
lm2=LinearRegression()
Z=df[['normalized-losses' , 'highway-mpg']]
Y=df['price']
lm2.fit(Z,Y)
Yhat=lm2.predict(Z)
print(Yhat[0:4])
print("The intercept is ", lm2.intercept_)
print("The slope is ", lm2.coef_)

# visualization
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#plt.show()
plt.close()

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#plt.show()
plt.close()

# weak correlation
print(df[['peak-rpm','highway-mpg', 'price']].corr())

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
#plt.show()
plt.close()

##
# Multiple Linear Regression
print("\n\nModel 5: Using horsepower, curb-weight, engine-size, highway-mpg as predictors")
# MLR
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.kdeplot(df['price'], color="r", label="Actual Value")
sns.kdeplot(Y_hat, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

# Polynomial Regression and Pipelines
print("\n\nPolynomial Regression and Pipelines")
x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')

f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1, x, y, 'highway-mpg')

# Use a polynomial of the 11th order   
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
print("Z-shape ",Z.shape)
print("Z_pr-shape ", Z_pr.shape)

#
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
print(pipe)

Z = Z.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

## Write your code below and press Shift+Enter to execute 
Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

print("\n\nPrediction using pipeline")
#
# checking the data
##highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-squared is: ', lm.score(Z, df['price']))
# calculate the MSE
Y_predict_multifit = lm.predict(Z)
# compare the predicted results with the actual results
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

# check r-squared
r_squared = r2_score(y, p(x))
print('The R-squared value is: ', r_squared)
# calculate the MSE
mean_squared_error(df['price'], p(x))
