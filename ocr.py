#!/usr/bin/env python

# this code exists to help us visually inspect the images
DEBUG = True

if DEBUG:
	# PIL = python library that lets you write & read .png images
	from PIL import Image
	import numpy as np

	# reads an image
	def read_image(path):
		return np.asarray(Image.open(path).convert('L')) # L is the format for bytes/pixels

	# saves an image as a png
	def write_image(image,path):
		img = Image.fromarray(np.array(image),'L')
		img.save(path)



# paths
DATA_DIR = 'data/'
TEST_DIR = 'test/' # png images from a data set
# sets: mnist(digit dataset), fashion_mnist(clothes dataset)
DATA_SET = 'fashion_mnist'

# data=images, labels=true values
TEST_DATA_FILENAME = DATA_DIR + DATA_SET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATA_SET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATA_SET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATA_SET + '/train-labels-idx1-ubyte'



# converts a byte to an int
def bytes_to_int(byte_data):
	return int.from_bytes(byte_data, 'big')



# reads files
# formats are taken from mnist's website
# images are in binary format
def read_images(filename, nr_max_images=None):
	images = []
	# rb=read binary
	with open(filename, 'rb') as f:
		not_important = f.read(4) # 4 bytes ~ magic number
		nr_images = bytes_to_int(f.read(4)) # number of images

		# to avoid reading a lot of images
		if nr_max_images:
			nr_images = nr_max_images

		nr_rows = bytes_to_int(f.read(4)) # number of rows
		nr_columns = bytes_to_int(f.read(4)) # number of columns
		# you cannot iterate through bytes
		for image_index in range(nr_images):
			image = []
			for row_index in range(nr_rows):
				row = []
				for column_index in range(nr_columns):
					pixel = f.read(1)
					row.append(pixel)
				image.append(row)
			images.append(image)
	return images



# reads labels
def read_labels(filename, nr_max_labels=None):
	labels = []
	# rb=read binary
	with open(filename, 'rb') as f:
		not_important = f.read(4) # 4 bytes ~ magic number
		nr_labels = bytes_to_int(f.read(4)) # number of images

		# to avoid reading a lot of images
		if nr_max_labels:
			nr_labels = nr_max_labels

		# you cannot iterate through bytes
		for label_index in range(nr_labels):
			label = bytes_to_int(f.read(1))
			labels.append(label)
	return labels



# flattens a list of lists
# for sublist in l:
#	for pixel in sublist:
#		return pixel
def flatten_list(l):
	return [pixel for sublist in l for pixel in sublist]



# returns a list of features
# images=[[first_row],[second_row]...]
# but we need:
# [pixel, pixel...]
# X = data set (set of samples)
def extract_features(X):
	return [flatten_list(sample) for sample in X]



# Euclidean distance
# x-image
# y-image
def dist(x,y):
	# zip:
	# x=[1,2,3]
	# y=['a','b','c']
	# => [[1,'a'],[2,'b'],[3,'c']]
	return sum(
	[
		(bytes_to_int(x_i) - bytes_to_int(y_i))**2
		for x_i,y_i in zip(x,y)]
	)**(0.5)



# for every training digit we're going to find the distance between that and our testing digit
def get_training_distances_for_test_sample(X_train, test_sample):
	return [dist(train_sample, test_sample) for train_sample in X_train]


# l=[1,1,2] => l.count=[2,2,1]
def get_most_frequent_element(l):
	return max(l, key=l.count)



# knn = k-nearest neighbors algorithm ~ "the lazy algorithm"
# STEPS:
# 1. select the number k of the neighbors
# 2. calculate the Euclidean distance of knn
# 3. take the k nearest neighbors as per calculated Euclidean distance
# 4. among these k neighbors, count the number of the data points in each category
# 5. assign the new data points to that category for which the number of the neighbor is maximum
# 6. the model is ready!
# k should be an odd number!! 5 is the best value
def knn(X_train, y_train, X_test, k=3):
	y_pred = [] # predicted labels
	# sample = data points
	for test_sample in X_test:
		training_distances = get_training_distances_for_test_sample(X_train, test_sample)
		# we want the indices of the closest elements
		# enumerate: [[0,'x'],[1,'a']...] - list of tuples
		sorted_distance_indices = [
			pair[0]
			for pair in sorted(enumerate(training_distances), key=lambda x:x[1])
		]
		# we want the k smallest elements
		candidates = [
			y_train[idx]
			for idx in sorted_distance_indices[:k]
		]
		# we want to find the candidate that occurs the most
		top_candidate=get_most_frequent_element(candidates)
		y_pred.append(top_candidate)
	return y_pred



# returns the garmet with a label from the list
def get_garment_from_label(label):
	return [
		'T-shirt/top',
		'Trouser',
		'Pullover',
		'Dress',
		'Coat',
		'Sandal',
		'Shirt',
		'Sneaker',
		'Bag',
		'Ankle boot'
	][label]



def main():

	n_train = 1000
	n_test = 10
	k = 7

	print(f'Data set: {DATA_SET}')
	print(f'n_train: {n_train}')
	print(f'n_test: {n_test}')
	print(f'k: {k}')

	# convention: X = data set; y = labels
	X_train = read_images(TRAIN_DATA_FILENAME,n_train)
	y_train = read_labels(TRAIN_LABELS_FILENAME,n_train)
	X_test = read_images(TEST_DATA_FILENAME,n_test)

	# you don't normally have these values
	# (they are not even passed to the knn method)
	# we have them here to see if our program works correctly
	y_test = read_labels(TEST_LABELS_FILENAME,n_test)

	if DEBUG:
		for idx,test_sample in enumerate(X_test):
			write_image(test_sample,f'{TEST_DIR}{idx}.png')

		# my test
		# X_test = [read_image(f'{DATA_DIR}our_test.png')]
		# y_test = [1]

	X_train = extract_features(X_train)
	X_test = extract_features(X_test)

	y_pred = knn(X_train, y_train, X_test, k)

	# the accuracy
	accuracy = sum([
		int(y_pred_i == y_test_i)
		for y_pred_i, y_test_i
		in zip(y_pred, y_test)
	]) / len(y_test)

	if DATA_SET == 'fashion_mnist':
		garments_pred = [
				get_garment_from_label(label)
				for label in y_pred
				]
		print(f'Predicted garments: {garments_pred}')
	else:
		print(f'Predicted labels: {y_pred}')

	print(f'Accuracy: {accuracy * 100}%')


if __name__ == '__main__':
	main()
