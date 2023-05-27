"""
Represents a single activation layer in the neural network. These 
layers convert the output from a dense layer into a value between
0 and 1 for the next dense layer using some activation function,
and a value for backpropagation using the activation_prime function.
"""

import numpy as np
from layer import Layer

class Activation(Layer):

  def __init__(self, activation, activation_prime):
    self.activation = activation
    self.activation_prime = activation_prime

  # Applies the activation function to a given input layer,
  # returning the layer with values mapped to [0,1].
  def forward(self, input):
    self.input = input
    return self.activation(self.input)
  
  # Applies the prime of the activation function to a given
  # output gradient. This is used to correct input values
  # during the back propagation phase.
  def backward(self, output_gradient, learning_rate):

    # Note the use of multiply and not dot product here...
    # This is required in order to pass the correct matrix
    # of values back during backpropagation.
    return np.multiply(output_gradient, self.activation_prime(self.input))