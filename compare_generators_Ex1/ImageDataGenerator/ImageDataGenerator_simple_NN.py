import sys
import getopt
import random
import os
import numpy as np
import keras
from keras import backend as K

### Helper:
def usage():
	print('\nUsage: python3 ImageDataGenerator_simple_NN.py [options] [path_to_dataset_info_file]')
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

### ImageDataGenerator that reads images directly from the disk and resize them to 32x32:
def setup_regular_generators(train_dir, validation_dir, batch_size, seed):
	# Set up train generator:
	train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(directory=train_dir,
														color_mode='rgb',
														target_size=(32, 32),
														batch_size=batch_size,
														class_mode='categorical',
														shuffle=True,
														seed=seed)
	# Set up validation generator:
	validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
																  color_mode='rgb',
																  target_size=(32, 32),
																  batch_size=batch_size,
																  class_mode='categorical',
																  shuffle=False,
																  seed=seed)
	return train_generator, validation_generator

### Minimal architecture:
def load_simple_model_keras(num_classes):
	simple_model = keras.models.Sequential()
	simple_model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
	simple_model.add(keras.layers.Dense(num_classes, activation='softmax'))
	return simple_model

### Test ImageDataGenerator on a simple architecture:
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
	path2info_file = args[0]
	# Parse dataset information:
	num_train_samples, num_val_samples, _, num_classes = parse_dataset_info(path2info_file)
	# Number of steps per epoch:
	steps_per_epoch = int(np.floor(num_train_samples / float(batch_size)))
	# Validation steps:
	validation_steps = int(np.floor(num_val_samples / float(batch_size)))
	# Fix randomness:
	random.seed = 951
	# Open a session to perform training:
	K.get_session()
	# Load a simple architecture:
	simple_model = load_simple_model_keras(num_classes)
	simple_model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
						 loss='categorical_crossentropy',
						 metrics=['accuracy'])
	simple_model.summary()	
	# Set up train and validation generators:
	train_dir = "./split_resized_images/train"
	validation_dir = "./split_resized_images/val"
	train_generator, validation_generator = setup_regular_generators(train_dir, validation_dir, batch_size, random.seed)
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
