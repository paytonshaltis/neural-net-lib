"""
Contains the main train() method that uses the backpropagation
and training algorithms to tune the neural network.
"""

import numpy as np
import math

# The main prediction method. Used during the training process, though
# will primarily be used for making predictions after a neural network
# has been trained.
def predict(network, input):

  # The output initially starts as the input to this layer.
  output = input

  # Apply the forward() function for all layers in the network,
  # all the way through to the final output layer.
  for layer in network:
    output = layer.forward(output)

  # Return this value as the prediction for the current input.
  return output

# The main training method. See comments below for parameter descriptions.
# Will modify each layer's attribute data with the tuned weights and biases
# upon completion of training.
def train(
        network,              # The list of all layers to make up a network.
        loss, loss_prime,     # The loss and loss_prime functions used.
        x_train, y_train,     # The input and output matrices of training data.
        epochs = 1000,        # The number of iterations of training.
        learning_rate = 0.01, # The advanced or decreased learning rate.
        verbose = True,       # Prints error / cost values during execution.
        updates = False):     # Prints updates on the network's progress.     
  
  # Use epochs to dictate the number of training iterations.
  percentage = 0
  for iteration in range(epochs):
    error = 0

    # For each pair of input and output matrices:
    for x, y in zip(x_train, y_train):

      # Retrieve the prediction for the current input. Does a full
      # traversal of the network using this input of the 0th layer.
      output = predict(network, x)

      # Compute the error for display purposes ONLY; otherwise, the
      # prime version of the function and backpropagation are sufficient.
      error += loss(y, output) if verbose else 0

      # Perform the backpropagation for the current input. Once a 
      # gradient is determined, backward() for the current input can
      # be done for all layers of the network.
      gradient = loss_prime(y, output)
      for layer in reversed(network):
        gradient = layer.backward(gradient, learning_rate)

      # If updates is marked as true, print a percentage of completion
      # at each 1% interval.
      if updates:
        new_percentage = math.floor((iteration / epochs) * 100)
        if(new_percentage > percentage):
          percentage = new_percentage
          print(f"Training {percentage}% complete...")

    # Finally, compute and print the error for each epoch if verbose
    # was marked as true.
    if verbose:
      error /= len(x_train)
      print(f"Epoch {iteration + 1}/{epochs}, Error={error}")


# Returns the accuracy of a network using the provided input and labels,
# returning a number in the range [0,1].
def acc(network, input, labels):
  correct = 0
  for x, y in zip(input, labels):
    correct += 1 if np.argmax(predict(network, np.reshape(x, (28 * 28, 1)))) == y else 0
  return correct / len(labels)