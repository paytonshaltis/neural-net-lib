"""
Sample code for training a neural network on the MNIST dataset of
hand-written numbers.
"""

import numpy as np
import math

# Prints out a 28x28 image of the given hand-written letter. Uses
# various ASCII characters to alter the lighting values for each pixel.
def print_mnist_image(image):
  PX_CHARS = [".", "|", "-", "+", "X", "&", "%", "$", "#", "@"]
  for row in image:
    for pixel in row:
      print(f"{PX_CHARS[math.floor(pixel / 25.6)]}", end=' ')
    print()
  

# Load the data from the MNIST dataset.
data = np.load('./data/mnist.npz')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

print_mnist_image(x_train[1])