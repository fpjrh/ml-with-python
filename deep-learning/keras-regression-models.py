import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# find concrete data set 

# Create a model
model = Sequential()

n_cols = concrete_data.shape[1] 

model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(concrete_data, concrete_target, batch_size=1, epochs=100, verbose=1)

predictions = model.predict(new_data)
print(predictions)

# use keras for classification

# Imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Create a model
model = Sequential()

n_cols = car_data.shape[1]
targets = to_categorical(car_data['class'])

model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(car_data, targets, batch_size=1, epochs=100, verbose=1)

predictions = model.predict(new_data)
print(predictions)
