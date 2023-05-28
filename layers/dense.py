"""
Represents a single dense layer in the neural network. This layer
has a number of inputs, outputs, and implemented methods for computing
the forward and backward values. This class extends functionality
of the simple base class Layer.
"""

import numpy as np
from layers.layer import Layer

class Dense(Layer):
  
  # Initialize weights and biases with random values.
  def __init__(self, input_size, output_size):
    
    # The weights attribute should be a matrix.
    # The biases attribute should be a column vector.  
    self.weights = np.random.randn(output_size, input_size)
    self.biases = np.random.randn(output_size, 1)

  # Returns a matrix of outputs for this dense layer.
  # Takes input as param for repeated application of the 
  # function on this layer from previous layers.
  def forward(self, input):
    self.input = input

    # Simple weight and bias application function for
    # each combination of inputs and weights, plus the biases
    # designated for each node for the layer.
    return np.dot(self.weights, self.input) + self.biases
  
  # Returns an input gradient for use during the backpropagation
  # algorithm of the neural network. Takes output_gradient as
  # a param for repeated application of the function on this 
  # layer from later layers.
  def backward(self, output_gradient, learning_rate):

    # Calculations for the gradients for each tunable parameter
    # during backpropagation. Note that each of the gradients used are
    # simplified expressions of the derivative of the error with 
    # respect to (1) weights, (2) biases, and (3) inputs.
    weights_gradient = np.dot(output_gradient, self.input.T)
    biases_gradient = output_gradient
    input_gradient = np.dot(self.weights.T, output_gradient)

    # Modifies the weights and biases based on both the gradient
    # and the learning rate parameter. This is where the error from
    # the final layer propagates its gradient backwards through the 
    # entire neural network.
    self.weights -= (learning_rate * weights_gradient)
    self.biases -= (learning_rate * biases_gradient)

    # Return the calculated input_gradient for the previous
    # layer to use for its backpropagation.
    return input_gradient
