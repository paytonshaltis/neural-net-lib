"""
Sample code for the neural network library. Demonstrates how simple
Numpy arrays can be used as input for a defined network in order to train
the weights and biases for the network.

Specifically, this neural network is set up to calculate XOR for a given
set of bits. It uses the four possible combinations to tune a neural
network and ensure that all is working correctly.

XOR definition:
| Bit 1 | Bit 2 | Output |
|-------|-------|--------|
|   0   |   0   |   0    |
|   0   |   1   |   1    |
|   1   |   0   |   1    |
|   1   |   1   |   0    |
"""

import sys
sys.path.append("..")

import numpy as np
from activations.tanh import Tanh
from dense import Dense
from mse import mse, mse_prime
from network import train, predict

# Define the input arrays. For training data, a matrix of data and a 
# matrix of labels are required. We use np.reshape in order to convert
# the standard Python array into the properly-sized matrix. Note that
# despite the labels only containing a single piece of data, each must
# technically be a 1x1 matrix.
DATA = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

LABELS = [
  [0],
  [1],
  [1],
  [0]
]

X = np.reshape(DATA, (4, 2, 1))
Y = np.reshape(LABELS, (4, 1, 1))

# The network can contain any number of layers, but the layer setup must
# conform to the following rules:
#  1. The first layer must have the same number of inputs as each data
#     point. Because each data point is a pair of bits, the first layer's
#     input must be of size 2.
#  2. Each layer's input must match the previous layer's output, and each
#     layer's output must match the next layer's input.
#  3. The final layer must have the same number of outputs as each label
#     point. Because each label point is a single number, the final layer's
#     output must be of size 1.
#  4. Each dense layer must be followed by a layer that extends the
#     Activation class (example: Tanh(), Sig(), etc.).
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# Finally, call the train() function with the proper pieces of training
# data, labels, error functions, epochs, learning rates, and verbosity.
train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1, verbose=False, updates=False)

# Once the network has received some training, we can pass some 'untrained'
# data in an analyze the values of the final layer. We should see almost complete
# confidence that each pair is or isn't true or false in XOR.
UNTRAINED = [
  [1, 0],     # 0.99
  [0, 0],     # 0.00
  [0, 1],     # 0.99
  [1, 1]      # 0.00
]

for ut in np.resize(UNTRAINED, (4, 2, 1)):
  print(f"Prediction for [{ut[0][0]},{ut[1][0]}]: {abs(round(predict(network, ut)[0][0], 2))}")