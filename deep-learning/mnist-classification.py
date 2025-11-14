import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist

# define classification model
def classification_model(num_pixels):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

# Display the first image in the training set
#plt.imshow(X_train[0])
#plt.show()

# flatten images to one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print("Number of classes: ", num_classes)

# Build the model
model = classification_model(num_pixels)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)
#evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
# print the accuracy
print("Accuracy: {} \n Error: {}".format(scores[1], 1-scores[1]))

# Let's see how well the model generalizes
predictions = model.predict(X_test)
print("The predictions are: \n", predictions)
