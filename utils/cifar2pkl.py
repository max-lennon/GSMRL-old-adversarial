import argparse
import os
import numpy as np
import pickle as pkl
import torchvision as tv #pytorch 1.4.0, torchvision 0.5.0
# from tensorflow.keras.datasets import mnist

## TODO: Add comments for updated functions


def load_cifar_dset (filepath, download, val_split):
	cifar_train = tv.datasets.CIFAR10(filepath, download=download) # Use the torchvision function to get values from MNIST. write "download=True" as an argument if needed, but I am assuming you have the file for versatility's sake.
	train_X = []
	train_Y = []
	for _, (x, y) in enumerate(cifar_train):
		train_X.append(np.array(x)) # Convert the MNIST value to numpy.
		train_Y.append(np.array(y))
		
	split_point = int(len(train_Y) * val_split)
	val_X = train_X[split_point:]
	val_Y = train_Y[split_point:]
	
	train_X = train_X[:split_point]
	train_Y = train_Y[:split_point]
		
	cifar_test = tv.datasets.CIFAR10(filepath, train=False, download=download)
	test_X = []
	test_Y = []
	for _, (x, y) in enumerate(cifar_test):
		test_X.append(np.array(x)) # Convert the MNIST value to numpy.
		test_Y.append(np.array(y))
	return (np.array(train_X), np.array(train_Y)), (np.array(val_X), np.array(val_Y)), (np.array(test_X), np.array(test_Y)) # Return the numpy value.


def pickle_dict(train_data, val_data, test_data, filepath, filename):
	complete_path = os.path.join(filepath, filename) # Concatenate the filepath and filename to get the file destination.
	
	dset_dict = {"train": train_data, "valid": val_data, "test": test_data}

	with open(complete_path, 'wb') as f: # open the file for writing.
		pkl.dump(dset_dict, f, pkl.HIGHEST_PROTOCOL) # dump the pkl values.

def cifar2pkl(cifarFilepath, pklFilepath, pklFilename, download=False, val_split=0.8):
	train, val, test = load_cifar_dset(cifarFilepath, download=download, val_split=val_split) # Run the mnist2numpy function to get the numpy value.
	pickle_dict(train, val, test, pklFilepath, pklFilename) # Use the numpy value with the numpy2pkl function to get the pkl file.


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("save_path")
	args = parser.parse_args()
	cifar2pkl("datasets/cifar/", args.save_path, "cifar.pkl", download=True)