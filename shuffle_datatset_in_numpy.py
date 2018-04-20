import numpy as np

def shuffle_x_y(x,y):
	# Assumption x and y are numpy arrays
	data_size = x.shape[0]
	shuffle_indices = np.random.permutation(np.arange(data_size))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]
	return (x_shuffled,y_shuffled)
	
