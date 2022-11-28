import argparse
import os
import numpy as np
import pickle as pkl
import torchvision as tv #pytorch 1.4.0, torchvision 0.5.0
# from tensorflow.keras.datasets import mnist

## TODO: Add comments for updated functions


def load_mnist_dset (filepath, download):
	mnist_train = tv.datasets.MNIST(filepath, download=download) # Use the torchvision function to get values from MNIST. write "download=True" as an argument if needed, but I am assuming you have the file for versatility's sake.
	train_X = []
	train_Y = []
	for _, (x, y) in enumerate(mnist_train):
		train_X.append(np.array(x)) # Convert the MNIST value to numpy.
		train_Y.append(np.array(y))
		
	mnist_val = tv.datasets.MNIST(filepath, train=False, download=download)
	val_X = []
	val_Y = []
	for _, (x, y) in enumerate(mnist_val):
		val_X.append(np.array(x)) # Convert the MNIST value to numpy.
		val_Y.append(np.array(y))
		
	return (np.array(train_X), np.array(train_Y)), (np.array(val_X), np.array(val_Y)) # Return the numpy value.


def pickle_dict(train_data, val_data, filepath, filename):
	complete_path = os.path.join(filepath, filename) # Concatenate the filepath and filename to get the file destination.
	
	dset_dict = {"train": train_data, "valid": val_data}
	
	with open(complete_path, 'wb') as f: # open the file for writing.
		pkl.dump(dset_dict, f, pkl.HIGHEST_PROTOCOL) # dump the pkl values.

def mnist2pkl(mnistFilepath, pklFilepath, pklFilename, download=False):
	train, val = load_mnist_dset(mnistFilepath, download=download) # Run the mnist2numpy function to get the numpy value.
	pickle_dict(train, val, pklFilepath, pklFilename) # Use the numpy value with the numpy2pkl function to get the pkl file.


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("save_path")
	args = parser.parse_args()
	mnist2pkl("datasets/mnist/", args.save_path, "mnist.pkl", download=True)
