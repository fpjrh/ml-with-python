import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

FILENAME = '/Users/fpj/Development/python/ml-with-python/k-nearest/data/teleCust1000t.csv'

df=pd.read_csv(FILENAME)
print(df.head(10))
# let's see how many of each type of customer there are
print(df['custcat'].value_counts())
# lets plot this
#df.hist(column='income', bins=50)
#plt.show()
# it looks like there are some outliers in this data set, lets remove them
print(df.columns)
# convert the pandas dataframe to a numpy array
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
print(X[0:5])
#
y = df['custcat'].values
print(y[0:5])
# normalize the data attributes
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)
# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)
# make predictions
yhat = neigh.predict(X_test)
print(yhat[0:5])
# accuracy evaluation
from sklearn import metrics
print("Train set Accuracy (k=4): ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: (k=4)", metrics.accuracy_score(y_test, yhat))
# lets try with k=6
k = 6
neigh6 = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# make predictions
yhat6 = neigh6.predict(X_test)
# accuracy evaluation
print("Train set Accuracy (k=6): ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: (k=6)", metrics.accuracy_score(y_test, yhat6))
# find the best k
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    print(n, mean_acc[n-1])
print(mean_acc)
# plot the results
k_range = range(1, Ks)
plt.plot(k_range, mean_acc, 'g')
plt.fill_between(k_range, mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(k_range, mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()
#
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 