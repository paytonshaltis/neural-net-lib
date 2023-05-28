"""
Represents a specific set of activation and activation_prime
functions for mapping layer outputs to the range [-1,1]. This
specific activation function uses the hyperbolic tangent function
and its derivative. Note that sigmoid, another common activation
function, maps its range to [0,1] instead.
"""

import numpy as np
from layers.activation import Activation

class Tanh(Activation):

  # Defines the tanh function and its derivative.
  def __init__(self):
    tanh = lambda x: np.tanh(x)
    tanh_prime = lambda x: 1 - np.tanh(x) ** 2

    # Call the constructor of the base Activation class in
    # order to assign these two functions to the object.
    super().__init__(tanh, tanh_prime)