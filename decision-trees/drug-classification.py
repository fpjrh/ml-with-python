import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

FILEPATH = '/Users/fpj/Development/python/ml-with-python/decision-trees/data/drug200.csv'

# Step 1: Load the data
my_data = pd.read_csv(FILEPATH)
print(my_data[0:5])
# print the size of the dataframe
print(my_data.size)
print(my_data.shape)

# Step 2: Pre-processing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])
# Pre-process the data to convert categorical data into numerical data
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])
y = my_data["Drug"]
print(y[0:5])

# Step 3: Setting up the Decision Tree
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print("X training set shape ", X_trainset.shape)
print("y training set shape ", y_trainset.shape)

print("X test set shape ", X_testset.shape)
print("y test set shape ", y_testset.shape)

# Step 4: Training the model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)
drugTree.fit(X_trainset, y_trainset)

# Step 5: Prediction
predTree = drugTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

# Step 6: Evaluation
from sklearn import metrics
from matplotlib import pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

from sklearn.tree import export_graphviz

export_graphviz(drugTree, out_file='tree.dot', feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

with open('tree.dot') as f:
    dot_graph = f.read()
    print(dot_graph)
    f.close()
