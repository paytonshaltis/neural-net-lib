"""
Contains the functions for calculating the mean square error for
a neural network, as well as the derivative of this function.
"""

import numpy as np

# Calculates and returns the mean square error of the current
# network iteration's results. Note that the mean SQUARED value
# is used to emphasize desired changes in weights and biases
# for the next iteration of training.
def mse(y_true, y_pred):
  return np.mean(np.power(y_true - y_pred, 2))

# Used for backpropagation for the mse values from the final
# layer up until the first layer.
def mse_prime(y_true, y_pred):
  return 2 * (y_pred - y_true) / np.size(y_true)