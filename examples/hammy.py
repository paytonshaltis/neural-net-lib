"""
A more realistic application of neural networks: predicting whether or not 
Hammy the stuffed hamster is currently present in an image. Images will be
processed into grayscale, reduced-resolution images, and then fed into a
neural network. The network will be trained on these images with the goal
of feeding it a new image and having it predict whether or not Hammy is
present in the image.
"""

import sys
sys.path.append('..')

from PIL import Image
import numpy as np
import os

from network import train, predict
from layers.dense import Dense
from activations.sig import Sig
from losses.mse import mse, mse_prime

# Constants
RES = 100                       # Resolution of the images (RES x RES)
RAW_DIR = '../data/raw'         # Directory containing raw images
PROC_DIR = '../data/processed'  # Directory containing processed images
TEST_DIR = '../data/test'       # Directory containing test images
EPOCHS = 1000                   # Number of epochs to train for
LR = 0.05                       # Learning rate

# Preprocesses all images in the raw directory and saves them
# to the processed directory.
def preprocess(raw_dir, proc_dir):
  # Create the directory if it doesn't exist.
  if not os.path.exists(proc_dir):
    os.makedirs(proc_dir)

  # Resize and convert all images to grayscale.
  for file in os.listdir(raw_dir):
    img = Image.open(f"{raw_dir}/{file}")
    img = img.resize((RES, RES))
    img = img.convert('L')
    img.save(f"{proc_dir}/{file}")



# Main function.
def main():
  # Preprocess the images.
  preprocess(RAW_DIR, PROC_DIR)

  # Store the pixel data and labels for each image.
  images = []
  labels = []
  for file in os.listdir(PROC_DIR):
    img = Image.open(f"{PROC_DIR}/{file}")
    images.append(list(img.getdata()))
    labels.append([1] if 'yes' in file else [0])
  images = np.resize(images, (len(images), RES * RES, 1))
  labels = np.resize(labels, (len(labels), 1, 1))

  # Create the neural network.
  network = [
    Dense(RES * RES, 100),
    Sig(),
    Dense(100, 1),
    Sig()
  ]

  # Train the network.
  errors = train(network, mse, mse_prime, images, labels, epochs=EPOCHS, learning_rate=LR, verbose=True)

  # Test and print results for each image in the test directory.
  for file in os.listdir(TEST_DIR):
    img = Image.open(f"{TEST_DIR}/{file}")
    img = img.resize((RES, RES))
    img = img.convert('L')
    img = list(img.getdata())
    img = np.resize(img, (RES * RES, 1))
    output = predict(network, img)
    print(f"{file}:")
    print(f"  Hammy: {output[0][0] * 100:.2f}%")

  # Write the errors to a file.
  with open('../errors.txt', 'w') as f:
    for error in errors:
      f.write(f"{error}\n")


if __name__ == "__main__":
  main()