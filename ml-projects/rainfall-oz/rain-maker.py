# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

WDOFILE = '/Users/fpj/Development/python/ml-with-python/ml-projects/rainfall-oz/data/weatherOZ.csv'
# Load the data
df = pd.read_csv(WDOFILE)
print("Dataframe shape: ", df.shape)
print("Dataframe columns: \n", df.columns)
print("Dataframe head: \n", df.head(5))
#
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
print("\nTraining set shape: ", x_train.shape, y_train.shape)
# create and train a linear regression model
LinearReg = LinearRegression()
# Train the model
LinearReg.fit(x_train, y_train)
# make predictions
predictions = LinearReg.predict(x_test)
# use predictions and y_test to calculate the value of each metric
LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
LinearRegression_R2 = metrics.r2_score(y_test, predictions)
LinearRegression_JaccardIndex = metrics.jaccard_score(y_test, predictions)
LinearRegression_F1 = metrics.f1_score(y_test, predictions)
LinearRegression_Accuracy = metrics.accuracy_score(y_test, predictions)
# print the metrics
print("Linear Regression MAE: ", LinearRegression_MAE)
print("Linear Regression MSE: ", LinearRegression_MSE)
print("Linear Regression RMSE: ", LinearRegression_RMSE)
print("Linear Regression R2: ", LinearRegression_R2)

# show the MAE, MSE and R2 in a tabular format using data frame for the linear regression model
Report = pd.DataFrame({'Metric': ['MAE', 'MSE', 'R2'], 'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]})
print("\nLinear Regression Metrics: \n", Report)

# Create a KNN model with n_neighbors = 4
KNN = KNeighborsClassifier(n_neighbors=4)
# Train the model
KNN.fit(x_train, y_train)
# make predictions
predictions = KNN.predict(x_test)
# use predictions and y_test to calculate the value of each metric
KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_Precision = metrics.precision_score(y_test, predictions)
KNN_Recall = metrics.recall_score(y_test, predictions)
KNN_F1_Score = metrics.f1_score(y_test, predictions)
# calculate the jaccaard score
KNN_JaccardIndex = metrics.jaccard_score(y_test, predictions)
# print the metrics
print("\nKNN Accuracy: ", KNN_Accuracy_Score)
print("KNN Precision: ", KNN_Precision)
print("KNN Recall: ", KNN_Recall)
print("KNN F1: ", KNN_F1_Score)
print("KNN Jaccard Index: ", KNN_JaccardIndex)
# show the accuracy, precision, recall, f1 and jaccard index in a tabular format using data frame for the KNN model
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [KNN_Accuracy_Score, KNN_F1_Score, KNN_JaccardIndex]})
print("\nKNN Metrics: \n", Report)

# Create a Decision Tree model with max_depth = 4
Tree = DecisionTreeClassifier(max_depth=4)
# train the model
Tree.fit(x_train, y_train)
# make predictions
predictions = Tree.predict(x_test)
# use predictions and y_test to calculate the value of Accuracy, Jaccard Index, F1 Score
Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
Tree_Precision = metrics.precision_score(y_test, predictions)
Tree_Recall = metrics.recall_score(y_test, predictions)
Tree_F1_Score = metrics.f1_score(y_test, predictions)
Tree_JaccardIndex = metrics.jaccard_score(y_test, predictions)
# print the metrics
print("\nDecision Tree Accuracy: ", Tree_Accuracy_Score)
print("Decision Tree Precision: ", Tree_Precision)
print("Decision Tree Recall: ", Tree_Recall)
print("Decision Tree F1 Score: ", Tree_F1_Score)
print("Decision Tree Jaccard Index: ", Tree_JaccardIndex)
# show the accuracy, precision, recall, f1 and jaccard index in a tabular format using data frame for the Decision Tree model
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [Tree_Accuracy_Score, Tree_F1_Score, Tree_JaccardIndex]})

print("\nDecision Tree Metrics: \n", Report)

# Use train_test_split to split the features and Y dataframes with a test_size of 0.2 and random_state of 1 
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)
# create and train a logistic regression model with the solver parameter set to 'liblinear'
LR = LogisticRegression(C=0.01, solver='liblinear')
# Train the model
LR.fit(x_train, y_train)
# make predictions with the predict and predict_proba methods
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)
# use predictions, predict_proba and y_test to calculate the value of each metric
LR_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
LR_Precision = metrics.precision_score(y_test, predictions)
LR_Recall = metrics.recall_score(y_test, predictions)
LR_F1_Score = metrics.f1_score(y_test, predictions)
LR_JaccardIndex = metrics.jaccard_score(y_test, predictions)
LR_Log_Loss = metrics.log_loss(y_test, predict_proba)
# print the metrics
print("\nLogistic Regression Accuracy: ", LR_Accuracy_Score)
print("Logistic Regression Precision: ", LR_Precision)
print("Logistic Regression Recall: ", LR_Recall)
print("Logistic Regression F1: ", LR_F1_Score)
print("Logistic Regression Jaccard Index: ", LR_JaccardIndex)
print("Logistic Regression Log Loss: ", LR_Log_Loss)
# show the accuracy, precision, recall, f1, jaccard index and log loss in a tabular format using data frame for the Logistic Regression model
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index', 'Log Loss'], 'Value': [LR_Accuracy_Score, LR_F1_Score, LR_JaccardIndex, LR_Log_Loss]})
print("\nLogistic Regression Metrics: \n", Report)

# create and train a SVM model with a linear kernel
SVM = svm.SVC(kernel='linear')
# Train the model
SVM.fit(x_train, y_train)
# make predictions
predictions = SVM.predict(x_test)

# use predictions and y_test to calculate the value of Accuracy Score, Jaccard Index, F1 Score
SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_Precision = metrics.precision_score(y_test, predictions)
SVM_Recall = metrics.recall_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
# print the metrics
print("\nSVM Accuracy: ", SVM_Accuracy_Score)
print("SVM Precision: ", SVM_Precision)
print("SVM Recall: ", SVM_Recall)
print("SVM F1: ", SVM_F1_Score)
print("SVM Jaccard Index: ", SVM_JaccardIndex)
# show the accuracy, precision, recall, f1 and jaccard index in a tabular format using data frame for the SVM model
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [SVM_Accuracy_Score, SVM_F1_Score, SVM_JaccardIndex]})
print("\nSVM Metrics: \n", Report)

# create a single report for all the models
Report = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Jaccard Index', 'Log Loss'], 'Value': [LR_Accuracy_Score, LR_Precision, LR_Recall, LR_F1_Score, LR_JaccardIndex, LR_Log_Loss]})
print("\nAll Models Metrics: \n", Report)


# Reporting section
Report = pd.DataFrame({'Metric': ['MAE', 'MSE', 'R2'], 'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]})
print("\nLinear Regression Metrics: \n", Report)
#
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [KNN_Accuracy_Score, KNN_F1_Score, KNN_JaccardIndex]})
print("\nKNN Metrics: \n", Report)
#
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [Tree_Accuracy_Score, Tree_F1_Score, Tree_JaccardIndex]})
print("\nDecision Tree Metrics: \n", Report)
#
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index', 'Log Loss'], 'Value': [LR_Accuracy_Score, LR_F1_Score, LR_JaccardIndex, LR_Log_Loss]})
print("\nLogistic Regression Metrics: \n", Report)
#
Report = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Jaccard Index'], 'Value': [SVM_Accuracy_Score, SVM_F1_Score, SVM_JaccardIndex]})
print("\nSVM Metrics: \n", Report)
#
# Construct a single dataframe report
Report = pd.DataFrame({'Model':['Linear Regression', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
                       'Accuracy':[LinearRegression_Accuracy, KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score], 
                       'F1':[LinearRegression_F1, KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score], 
                       'Jaccard Index':[LinearRegression_JaccardIndex, KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex], 'Log Loss':[0, 0, 0, LR_Log_Loss, 0]})
print("\nAll Models Metrics: \n", Report)
