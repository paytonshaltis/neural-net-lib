"""
Represents a specific set of activation and activation_prime
functions for mapping layer outputs to the range [0,1]. This
specific activation function uses the sigmoid function and its 
derivative. Note that hyperbolic tangent, another common activation
function, maps its range to [-1,1] instead.
"""

import numpy as np
from layers.activation import Activation

class Sig(Activation):
  
  def __init__(self):
    # Defines the sig function and its derivative.
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(x):
        s = sigmoid(x)
        return s * (1 - s)

    # Call the constructor of the base Activation class in
    # order to assign these two functions to the object.
    super().__init__(sigmoid, sigmoid_prime)