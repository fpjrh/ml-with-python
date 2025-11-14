import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

file_name = '/Users/fpj/Development/python/ml-with-python/project-work/data/medical_insurance_dataset.csv'

# Step 1: read the data
df = pd.read_csv(file_name, header=None)
# apply headers
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers
print(df.head(10))
#
print(df.info())
print(df.dtypes)
# clean data
df.replace('?', np.nan, inplace=True)
# Step 2: a bit of data wrangling
#
# smoker is a categorical attribute, replace with most frequent entry
df.replace({'smoker': {'yes': 1, 'no': 0}}, inplace=True)
df.replace({'smoker': np.nan}, 0, inplace=True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype('float').mean(axis=0)
df.replace({'age': np.nan}, mean_age, inplace=True)

# Update data types
df[["age"]] = df[["age"]].astype("int")
# update the charges column to have no more than 2 decimal places
df[["charges"]] = np.round(df[["charges"]],2)
print(df.head(10))
# Step 3: Exploratory Data Analysis
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color": "red"})
plt.ylim(0,)
plt.show()
# box plot
sns.boxplot(x="smoker", y="charges", data=df)
plt.show()
# show the correlation matrix
print(df.corr())
#
# Step 4: Model Development
X = df[['smoker']]
Y = df['charges']
lm = LinearRegression()
lm.fit(X,Y)
print(lm.score(X, Y))
#
# definition of Y and lm remain same as used in last cell. 
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(Z,Y)
print(lm.score(Z, Y))
#
# Y and Z use the same values as defined in previous cells 
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(r2_score(Y,ypipe))
#
# Z and Y hold same values as in previous cells
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)
# x_train, x_test, y_train, y_test hold same values as in previous cells
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))
# x_train, x_test, y_train, y_test hold same values as in previous cells
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))