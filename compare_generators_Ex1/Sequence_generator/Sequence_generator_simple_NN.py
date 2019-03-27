import sys
import getopt
import random
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import Callback
from PIL import Image

### Sequence generator inspired from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class SequenceGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, classes_dict, n_classes, batch_size, num_samples, dim, n_channels, shuffle=True):
		'Initialization'
		self.list_IDs = list_IDs
		self.classes_dict = classes_dict
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.num_samples = num_samples
		self.dim = dim
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(self.num_samples / float(self.batch_size)))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch:
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs:
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data:
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size, self.n_classes))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):

			# Load sample:
			image_PIL = Image.open(ID)
			num_rows, num_cols = image_PIL.size
			# Resize if needed:
			if not num_rows == self.dim[0] or not num_cols == self.dim[1]:
				image_PIL = image_PIL.resize(size=self.dim, resample=Image.BILINEAR)
			image_np = np.array(image_PIL)
			# Add a channel dimension if needed:
			if len(image_np.shape) == 2:
				image_np = np.expand_dims(image_np, axis=2)
			# Duplicate channels if needed:
			if image_np.shape[2] == 1 and not self.n_channels == 1:
				image_np = np.tile(image_np, (1, 1, self.n_channels))

			# Store image:
			X[i,] = image_np.astype('float32') / 255

			# Store label:
			label = ID.split('/')[-2]
			label_value = self.classes_dict[label]
			y[i] = keras.utils.to_categorical(label_value, num_classes=self.n_classes)

		return X, y

### Helper:
def usage():
	print('\nUsage: python3 Sequence_generator_simple_NN.py [options] [path_to_parsable_files]')
	print('Option list: -b/--batch_size, -e/--epochs, -h/--help, -m/--multi_processing, -w/--workers')

### Check whether input char is blank or not:
def is_blank_char(c):
	switcher = {' ': 1,
				'\t': 1, 
				'\n': 1,
				'\r': 1,
				'\f': 1,
				'\v': 1,
				'\a': 1,
				'\b': 1}
	default_non_blank_char = 0
	return switcher.get(c, default_non_blank_char)

### Extract the first word in a string. Words are separated by blank characters:
def get_first_word_in_string(s):
	begin_idx = 0
	end_idx = 0
	for idx, char in enumerate(s):
		if not is_blank_char(char):
			begin_idx = idx
			break
	for idx, char in enumerate(s[begin_idx:],begin_idx):
		if is_blank_char(char):
			end_idx = idx
			break
	return s[begin_idx:end_idx], begin_idx, end_idx

### Extract the first integer in a string.
def get_first_int_in_string(s):
	begin_idx = 0
	end_idx = 0
	for idx, char in enumerate(s):
		if char.isdigit():
			begin_idx = idx
			break
	for idx, char in enumerate(s[begin_idx:],begin_idx):
		if not char.isdigit():
			end_idx = idx
			break
	return int(s[begin_idx:end_idx]), begin_idx, end_idx

### Parse dataset informations:
def parse_dataset_info(path2info_file):
	info_types = ["num_train_samples", "num_val_samples", "num_test_samples", "num_classes"]
	info_values = dict(zip(info_types, ["", "", "", ""]))
	with open(path2info_file,'r') as info_file:
		lines = info_file.readlines()
		for line in lines:
			info_type, _, end_idx = get_first_word_in_string(line)
			info_type = info_type.split(":")[0]
			info_value, _, _ = get_first_int_in_string(line[end_idx:])
			info_values[info_type] = int(info_value)
	num_train_samples = info_values["num_train_samples"]
	num_val_samples = info_values["num_val_samples"]
	num_test_samples = info_values["num_test_samples"]
	num_classes = info_values["num_classes"]
	return num_train_samples, num_val_samples, num_test_samples, num_classes

### Classes correspondance obtained from a dictionary file:
def get_classes_dict(path_dictionary):
	classes_dict = {}
	with open(path_dictionary,'r') as dictionary:
		lines = dictionary.readlines()
		for line in lines:
			class_name, _, end_idx = get_first_word_in_string(line)
			class_value, _, _ = get_first_word_in_string(line[end_idx+2:])
			classes_dict[class_name] = int(class_value)
	return classes_dict

### Collect the paths to every images listed in a paths file:
def collect_images_dataset(paths_file):
	filenames = []
	with open(paths_file,'r') as paths:
		lines = paths.readlines()
		for line in lines:
			path, _, _ = get_first_word_in_string(line)
			filenames.append(path)
	filenames = [filename for filename in filenames if filename.endswith('.jpg')]
	return filenames

### Collect the paths to every train/val/test images, get dataset info and set up the classes dictionary:
def general_parser(paths2parsable_files):
	path_types = ["info_dataset", "dictionary", "train", "val", "test"]
	files_paths = dict(zip(path_types, ["", "", "", "", ""]))
	with open(paths2parsable_files,'r') as paths2parsable:
		lines = paths2parsable.readlines()
		for line in lines:
			path_type, _, end_idx = get_first_word_in_string(line)
			path_type = path_type.split(":")[0]
			path, _, _ = get_first_word_in_string(line[end_idx:])
			files_paths[path_type] = path
	num_train_samples, num_val_samples, num_test_samples, num_classes = parse_dataset_info(files_paths["info_dataset"])
	classes_dict = get_classes_dict(files_paths["dictionary"])
	train_filenames = collect_images_dataset(files_paths["train"])
	val_filenames = collect_images_dataset(files_paths["val"])
	test_filenames = collect_images_dataset(files_paths["test"])
	return num_train_samples, num_val_samples, num_test_samples, num_classes, classes_dict, train_filenames, val_filenames, test_filenames

### Minimal architecture:
def load_simple_model_keras(num_classes):
	simple_model = keras.models.Sequential()
	simple_model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
	simple_model.add(keras.layers.Dense(num_classes, activation='softmax'))
	return simple_model

### Test Sequence generator on a simple architecture:
def main(argv):
	try:                                
		opts, args = getopt.getopt(argv, "b:e:hmw:", ["batch_size=", "epochs=", "help", "multi_processing", "workers="])
	except getopt.GetoptError as error:  
		print(str(error))        
		usage()                         
		sys.exit(2) 
	## Default parameter values:    
	num_epochs = 5   
	batch_size = 64
	num_workers = 0
	multi_processing = False
	for opt, arg in opts:      
		if opt in ("-b", "--batch_size"):  
			batch_size = int(arg)
			assert batch_size > 0, "Batch size must be an int strictly larger than 0 !"   
		elif opt in ("-e", "--epochs"):  
			num_epochs = int(arg)
			assert num_epochs > 0, "Number of epochs must be an int strictly larger than 0 !"
		elif opt in ("-h", "--help"):      
			usage()
			sys.exit(0)
		elif opt in ("-m", "--multi_processing"):      
			multi_processing = True
		elif opt in ("-w", "--workers"):  
			num_workers = int(arg)
			assert num_workers >= 0, "Number of workers must be a non-negative int !"
		else:      
			print("Unhandled option !")    
			usage()                         
			sys.exit(2)                
	if not len(args) == 1:
		print("Unique argument specifying path to a file that contains dataset informations is required !")
		usage()
		sys.exit(2)
	paths2parsable_files = args[0]
	# Fix randomness:
	random.seed = 951
	# Parse dataset informations, classes dictionary and images' filenames of the dataset:
	num_train_samples, num_val_samples, _, num_classes, classes_dict, train_filenames, val_filenames, test_filenames = general_parser(paths2parsable_files)
	# Number of steps per epoch:
	steps_per_epoch = int(np.floor(num_train_samples / float(batch_size)))
	# Validation steps:
	validation_steps = int(np.floor(num_val_samples / float(batch_size)))
	# Open a session to perform training:
	K.get_session()
	# Load a simple architecture:
	simple_model = load_simple_model_keras(num_classes)
	simple_model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
						 loss='categorical_crossentropy',
						 metrics=['accuracy'])
	simple_model.summary()	
	# Set up train and validation generators:
	train_generator = SequenceGenerator(train_filenames, 
										classes_dict, 
										n_classes=num_classes, 
										batch_size=batch_size, 
										num_samples=num_train_samples, 
										dim=(32, 32), 
										n_channels=3)
	validation_generator = SequenceGenerator(val_filenames, 
											 classes_dict,
											 n_classes=num_classes, 
											 batch_size=batch_size, 
											 num_samples=num_val_samples, 
											 dim=(32, 32), 
											 n_channels=3)	
	# Fit the model:
	simple_model.fit_generator(train_generator,
							   steps_per_epoch=steps_per_epoch,
							   epochs=num_epochs,
							   validation_data=validation_generator,
							   validation_steps=validation_steps,
							   use_multiprocessing=multi_processing,
							   workers=num_workers)
	# Clean the current session:
	K.clear_session()

### Passing command line to the main():	
if __name__== "__main__":
	verbose = False
	main(sys.argv[1:])
