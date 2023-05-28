"""
Represents a specific set of activation and activation_prime
functions for mapping layer outputs to the range [0,1]. This
specific activation function uses the RELU function and its 
derivative.
"""

import numpy as np
from activation import Activation

class Relu(Activation):
    
    def __init__(self):
      # Defines the RELU function and its derivative.
      relu = lambda x: np.maximum(0, x)
      relu_prime = lambda x: np.where(x > 0, 1, 0)
  
      # Call the constructor of the base Activation class in
      # order to assign these two functions to the object.
      super().__init__(relu, relu_prime)