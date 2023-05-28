"""
Sample code for training a neural network on the MNIST dataset of
hand-written numbers.
"""

import numpy as np
import math
from dense import Dense
from activations.sig import Sig
from mse import mse, mse_prime
from network import train, predict

# Prints out a 28x28 image of the given hand-written letter. Uses
# various ASCII characters to alter the lighting values for each pixel.
def print_mnist_image(image):
  PX_CHARS = [".", "|", "-", "+", "X", "&", "%", "$", "#", "@"]

  # Should behave the same for both 1D and 2D arrays.
  if(image.shape != (28 * 28, 1)):
    for row in image:
      for pixel in row:
        print(f"{PX_CHARS[math.floor(pixel / 25.6)]}", end=' ')
      print()
  else:
    for i in range(28 * 28):
      print(f"{PX_CHARS[math.floor(image[i] / 25.6)]}", end=' ')
      if((i + 1) % 28 == 0):
        print()

# Load the data from the MNIST dataset.
data = np.load('./data/mnist.npz')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

# Prepare 1000 training samples for the network.
X = np.reshape(x_train[:60000], (60000, 28 * 28, 1))

# The Y matrix must be a 10x1 matrix for each label. The value at the
# index of the label is set to 1, and all other values are set to 0.
Y = np.zeros((60000, 10, 1))
for i in range(60000):
  Y[i][y_train[i]] = 1

# Create a network with 2 layers of 16 hidden neurons each.
network = [
  Dense(28 * 28, 40),
  Sig(),
  Dense(40, 40),
  Sig(),
  Dense(40, 10),
  Sig()
]

# Train the network using the training data.
train(network, mse, mse_prime, X, Y, epochs=100, learning_rate=0.075, verbose=True, updates=False)

# While the user wants to continue, predict the value of a random
# test sample and print the image and the prediction.
while True:
  index = np.random.randint(0, len(x_test))
  prediction = predict(network, np.reshape(x_test[index], (28 * 28, 1)))
  print_mnist_image(np.reshape(x_test[index], (28 * 28, 1)))

  # Print all of the values in the prediction matrix.
  print(f"Prediction: {np.argmax(prediction)}")
  for i in range(10):
    print(f"{i}: {abs(round(prediction[i][0], 2))}")

  input("Press enter to continue...")