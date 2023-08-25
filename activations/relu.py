"""
Represents a specific set of activation and activation_prime
functions for mapping layer outputs to the range [0,1]. This
specific activation function uses the RELU function and its 
derivative.
"""

import numpy as np
from layers.activation import Activation

class Relu(Activation):
    
    def __init__(self):
      # Defines the RELU function and its derivative.
      def relu(x):
        return np.maximum(0, x)
      
      def relu_prime(x):
        return np.where(x > 0, 1, 0)
  
      # Call the constructor of the base Activation class in
      # order to assign these two functions to the object.
      super().__init__(relu, relu_prime)