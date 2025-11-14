from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# Load the data
FILE_NAME = '/Users/fpj/Development/python/ml-with-python/linear-regression/data/laptop_pricing_dataset.csv'

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
    plt.show()


df = pd.read_csv(FILE_NAME, header=0)
print(df.head())
# drop unnecessary columns
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
# divide the dataset into dependent and independent variables
y = df['Price']
x = df.drop('Price', axis=1)

print("\n\n Single Variable LRE with 90/10 train/test split\n\n")
# split the dataset into training and testing sets and reserve 10% of the data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
# create a single variable linear regression model using CPU_frequency
lre = LinearRegression()
lre.fit(x_train[['CPU_frequency']], y_train)
# print the R^2 value of this model
print('R^2 for SVLRE CPU_frequency (test) : ', lre.score(x_test[['CPU_frequency']], y_test))
print('R^2 for SVLRE CPU_frequency (train): ', lre.score(x_train[['CPU_frequency']], y_train))

# run a 4-fold cross validation on the model
scores = cross_val_score(lre, x_train[['CPU_frequency']], y_train, cv=4)
# print the mean value of the R^2 score and its standard deviation
print("Accuracy : %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))

print("\n\n Single Variable LRE with 50/50 train/test split\n\n")
# Do the same again with 50% reserved for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])
lre.fit(x_train[['CPU_frequency']], y_train)
print('R^2 for SVLRE CPU_frequency (test) : ', lre.score(x_test[['CPU_frequency']], y_test))
print('R^2 for SVLRE CPU_frequency (train): ', lre.score(x_train[['CPU_frequency']], y_train))
scores = cross_val_score(lre, x_train[['CPU_frequency']], y_train, cv=4)
print("Accuracy : %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))
#
Rsquared_test = []
order=[1, 2, 3, 4, 5]
#
for ndeg in order:
    pr=PolynomialFeatures(degree=ndeg)
    x_train_pr=pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr=pr.fit_transform(x_test[['CPU_frequency']])
    lre.fit(x_train_pr, y_train)
    Rsquared_test.append(lre.score(x_test_pr, y_test))
#
for ndeg in order:
    print("R^2 for degree ", ndeg, " is ", Rsquared_test[ndeg-1])
#
plt.plot(order, Rsquared_test)
plt.title('R^2 for different polynomial degrees')
plt.xlabel('Degree')
plt.ylabel('R^2')
plt.show()
# Ridge Regression
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pr=pr.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
#
Rsquared_test = []
Rsquared_train = []
Alpha = np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha) 
    RidgeModel.fit(x_train_pr, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsquared_test.append(test_score)
    Rsquared_train.append(train_score)
# Plot the R^2 values for both test and train sets against alpha
plt.figure(figsize=(10, 6))
plt.plot(Alpha, Rsquared_test, label='Test')
plt.plot(Alpha, Rsquared_train, label='Train')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.title('R^2 for different values of alpha')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Grid search for the best alpha
parameters1 = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
# create a Ridge regression object
rr = Ridge()
# create a GridSearchCV object
grid_search = GridSearchCV(rr, parameters1, cv=4, scoring='r2')
# fit the model
grid_search.fit(x_train_pr, y_train)
# print the best parameters
print("Best parameters: ", grid_search.best_params_)
# print the best score
print("Best score: ", grid_search.best_score_)
# print the best estimator
print("Best estimator: ", grid_search.best_estimator_)
# print the best R^2 score for the test set
print("Best R^2 score for test set: ", grid_search.score(x_test_pr, y_test))
# print the best R^2 score for the training set
print("Best R^2 score for training set: ", grid_search.score(x_train_pr, y_train))
# print the best cross-validation score
print("Best cross-validation score: ", grid_search.best_score_)
# print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

