"""
Represents the base class for all layers in the neural network.
For this reason, the forward and backward methods are not implemented.
All other complex layers must implement these methods.
"""

class Layer:
  
  def __init__(self):
    self.input = None
    self.output = None
    
  def forward(self, input):
    # Not implemented by the base class.
    pass

  def backward(self, output_gradient, learning_rate):
    # Not implemented by the base class.
    pass