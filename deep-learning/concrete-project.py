import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

CONCRETE_FILE = '/Users/fpj/Development/python/ml-with-python/deep-learning/data/concrete_data.csv'

#
# Define regression_model
def single_hidden_layer_model(n_cols):
    model = Sequential()
    input_shape = keras.Input(shape=(n_cols,))
    model.add(input_shape)
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Define regression_model
def three_hidden_layer_model(n_cols):
    model = Sequential()
    input_shape = keras.Input(shape=(n_cols,))
    model.add(input_shape)
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# read the concrete data set
concrete_data = pd.read_csv(CONCRETE_FILE)
print(concrete_data.head())

# let's see how many data points we have
print("Dataset shape: ", concrete_data.shape)
# Let's check the dataset for any missing values
print("Describe the dataset: ", concrete_data.describe())
print("Any missing values?: ", concrete_data.isnull().sum())

# Split the data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']
n_cols = predictors.shape[1]
print("Number of predictors: ", n_cols)

mse_list = []
#
for repetition in range(50):
    #
    print("Iteration: ", repetition + 1)
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    # Build the model
    model = single_hidden_layer_model(n_cols)           # create the model
    model.fit(X_train, y_train, epochs=50, verbose=0)   # fit the model with 50 epochs
    # evaluate using mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print("Normalized predictors head:\n", predictors_norm.head())

norm_mse_list = []
#
for repetition in range(50):
    #
    print("Iteration : ", repetition + 1)
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    # Build the model
    model = single_hidden_layer_model(n_cols)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    # evaluate using mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    norm_mse_list.append(mse)
#
cent_mse_list = []
#
for repetition in range(50):
    #
    print("Iteration : ", repetition + 1)
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    # Build the model
    model = single_hidden_layer_model(n_cols)
    model.fit(X_train, y_train, epochs=100, verbose=0)
    # evaluate using mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    cent_mse_list.append(mse)
#
three_cent_mse_list = []
#
for repetition in range(50):
    #
    print("Iteration : ", repetition + 1)
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    # Build the model
    model = three_hidden_layer_model(n_cols)
    model.fit(X_train, y_train, epochs=100, verbose=0)
    # evaluate using mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    three_cent_mse_list.append(mse)
#
print("\nMean squared error: ", np.mean(mse_list))
print("Standard deviation: ", np.std(mse_list))
#
print("\nNormalized Mean squared error: ", np.mean(norm_mse_list))
print("Normalized Standard deviation: ", np.std(norm_mse_list))
#
print("\n100-epoch Mean squared error: ", np.mean(cent_mse_list))
print("100-epoch Normalized Standard deviation: ", np.std(cent_mse_list))
#
print("\nThree-hidden-layer 100-epoch Mean squared error: ", np.mean(three_cent_mse_list))
print("Three-hidden-layer 100-epoch Normalized Standard deviation: ", np.std(three_cent_mse_list))
#
