import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

CONCRETE_FILE = '/Users/fpj/Development/python/ml-with-python/deep-learning/data/concrete_data.csv'

#
# Define regression_model
def regression_model(n_cols):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

concrete_data = pd.read_csv(CONCRETE_FILE)
print(concrete_data.head())

# let's see how many data points we have
print("Dataset shape: ", concrete_data.shape)
# Let's check the dataset for any missing values
print("Describe the dataset: ", concrete_data.describe())
print("Any missing values?: ", concrete_data.isnull().sum())

# Split the data set into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# let's do a quick sanity check of the predictors and the target
print("Predictors head: \n", predictors.head())
print("Target head: \n", target.head())

# Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print("Normalized predictors head:\n", predictors_norm.head())
n_cols = predictors_norm.shape[1]

# Build a neural network
model = regression_model(n_cols)
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Evaluate the model
loss = model.evaluate(predictors_norm, target, verbose=0)
print("The mean squared error for the model is: ", loss)
# Let's see how well the model generalizes
predictions = model.predict(predictors_norm)
print("The predictions are: \n", predictions)
