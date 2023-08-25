"""
A simple application that uses previous weather data from Trenton, NJ
to predict weather. Inputs include the previous day's high and low,
precipitation, and the date. The output is the predicted high and low
for the next day.
"""

# Constants.
EPOCHS = 100                # Number of epochs to train for.
LEARNING_RATE = 0.0001          # Learning rate for training.

import sys
sys.path.append('..')

import numpy as np

from activations.sig import Sig
from activations.relu import Relu
from activations.tanh import Tanh
from layers.layer import Layer
from layers.dense import Dense
from network import train, predict
from losses.mse import mse, mse_prime

# Get and process the input weather data. The output is a list of lists,
# where each of the inner lists contains the following data:
#   0: Month
#   1: Day
#   2: Precipitation
#   3: Snowfall
#   4: High temperature
#   5: Low temperature
def process_data(path):
  
  # Read in the CSV file.
  with open(path, 'r') as f:
    data = f.readlines()[1:]

    # Remove all double quotes.
    data = [line.replace('"', '') for line in data]

    # Split the data into a list of lists.
    data = [line.strip().split(',') for line in data]
    
    # Extract the relevant data.
    data = [[line[3][5:7], line[3][8:10], line[8], line[9], line[11], line[12]] for line in data]

    # Eliminate any data missing the high or low temperatures.
    data = [line for line in data if line[4] != '' and line[5] != '']

    # Add 0s to precipitation and snowfall if they are missing.
    data = [[line[0], line[1], line[2] if line[2] != '' else 0, line[3] if line[3] != '' else 0, line[4], line[5]] for line in data]

    # Convert all data to floats.
    return [[float(item) for item in line] for line in data]

# Main function.
def main():

  # Convert data into numpy vectors.
  data = process_data('../data/weather.csv')[:20000]
  x_train = np.delete(np.resize(np.array(data[:-1]), (len(data) - 1, 6, 1)), [], 1)
  y_train = np.delete(np.resize(np.array(data[1:]), (len(data) - 1, 6, 1)), [0, 1, 2, 3], 1)

  # Print the shapes of the training data.
  print(x_train.shape, y_train.shape)

  # Create the network.
  network = [
    Dense(6, 64),
    Sig(),
    Dense(64, 2),
  ]

  # Train the network.
  errors = train(network, mse, mse_prime, x_train, y_train, EPOCHS, LEARNING_RATE, verbose = True, updates = False)
  print(errors)

  test = [5, 28, 0, 0, 46, 77]
  print(predict(network, np.reshape(test, (6, 1))))

if __name__ == '__main__':
  main()