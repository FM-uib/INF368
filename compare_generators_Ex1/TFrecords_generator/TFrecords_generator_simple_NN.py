import sys
import getopt
import random
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import Callback

### Partly inspired from:
### https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36?fbclid=IwAR0j0eQwY-ead148sqBI3qP0oP8YNpV4XhT7NPyM5D-xNeuE0hm5lX676WQ
### https://github.com/keras-team/keras/blob/master/examples/mnist_tfrecord.py?fbclid=IwAR1jZPCm0hB37JEDEl0z1cMkzVqZXlGfHp2k2hta-_YENF9R-e2Iy2lQs8s
### https://gist.github.com/datlife/abfe263803691a8864b7a2d4f87c4ab8?fbclid=IwAR0j0eQwY-ead148sqBI3qP0oP8YNpV4XhT7NPyM5D-xNeuE0hm5lX676WQ
### https://github.com/tensorflow/tensorflow/issues/20059

if K.backend() != 'tensorflow':
	raise RuntimeError('TFrecords compatibility requires tensorflow backend')

### Truncates/pads a float f to n decimal places without rounding
def float2str_with_trunc(f, n):
	s = '%.12f' % f
	i, p, d = s.partition('.')
	return '.'.join([i, (d+'0'*n)[:n]])

### Needed to perform validation when feeding input tensors to .fit()
class EvaluateInputTensor(Callback):
	""" Validate a model which does not expect external numpy data during training.
	Keras does not expect external numpy data at training time, and thus cannot
	accept numpy arrays for validation when all of a Keras Model's
	`Input(input_tensor)` layers are provided an  `input_tensor` parameter,
	and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
	Instead, create a second model for validation which is also configured
	with input tensors and add it to the `EvaluateInputTensor` callback
	to perform validation.
	It is recommended that this callback be the first in the list of callbacks
	because it defines the validation variables required by many other callbacks,
	and Callbacks are made in order.
	# Arguments
		model: Keras model on which to call model.evaluate().
		steps: Integer or `None`.
			Total number of steps (batches of samples)
			before declaring the evaluation round finished.
			Ignored with the default value of `None`.
	"""

	def __init__(self, val_model, steps, metrics_prefix='val', verbose=True):
		# parameter of callbacks passed during initialization
		# pass evalation mode directly
		super(EvaluateInputTensor, self).__init__()
		self.val_model = val_model
		self.num_steps = steps
		self.verbose = verbose
		self.metrics_prefix = metrics_prefix

	def on_epoch_end(self, epoch, logs={}):
		# self.model is a property reference to the model being trained on (see Callback class)
		self.val_model.set_weights(self.model.get_weights())
		results = self.val_model.evaluate(None, None, steps=self.num_steps,
										  verbose=self.verbose)
		metrics_str = ''
		for result, name in zip(results, self.val_model.metrics_names):
			metric_name = self.metrics_prefix + '_' + name
			logs[metric_name] = result
			if self.verbose > 0:
				metrics_str = metrics_str + metric_name + ': ' + float2str_with_trunc(round(result, 4), 4) + ' '
		if self.verbose > 0:
			print(metrics_str)

### Helper:
def usage():
	print('\nUsage: python3 TFrecords_generator_simple_NN.py [options] [path_to_parsable_files]')
	print('Option list: -b/--batch_size, -e/--epochs, -h/--help')

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

### Useful to convert a tensor into a np_array:
def tensor2nparray(tensor):
	np_array_tensor = tf.Session().run(tensor)
	return np_array_tensor

### Collect the paths to every shards listed in a paths file:
def collect_TFrecords_dataset(paths_file):
	filenames = []
	with open(paths_file,'r') as paths:
		lines = paths.readlines()
		for line in lines:
			path, _, _ = get_first_word_in_string(line)
			filenames.append(path)
	filenames = [filename for filename in filenames if filename.endswith('.tfrecords')]
	return filenames

### Collect the paths to every train/val/test shards, get dataset info and set up the classes dictionary:
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
	train_filenames = collect_TFrecords_dataset(files_paths["train"])
	val_filenames = collect_TFrecords_dataset(files_paths["val"])
	test_filenames = collect_TFrecords_dataset(files_paths["test"])
	return num_train_samples, num_val_samples, num_test_samples, num_classes, classes_dict, train_filenames, val_filenames, test_filenames

### Normalize the values inside a tensor into [0,1], convert the values as float32:
def normalize_tensor(input_tensor):
	output_tensor = tf.math.divide(tf.cast(input_tensor, tf.float32), 
								   tf.constant(255.0))
	return output_tensor

### Decode an image and its features stored as an example inside a TFrecords:
def decode_example(ciphered_example):
	# Define the features of your TFrecords encoding:
	keys_to_features = {'height': tf.FixedLenFeature([], tf.int64),
						'width': tf.FixedLenFeature([], tf.int64),
						'depth': tf.FixedLenFeature([], tf.int64),
						'label': tf.FixedLenFeature([], tf.int64),
						'image_raw': tf.FixedLenFeature([], tf.string)}    
	# Decode example's features:
	example_features = tf.parse_single_example(ciphered_example, keys_to_features)
	# Turn image string into a tensor of uint8:
	tensor_image = tf.decode_raw(example_features['image_raw'], tf.uint8)
	# Store the image label:
	tensor_label = tf.cast(example_features['label'], tf.int32)
	# Store image dimensions:
	num_rows = tf.cast(example_features['height'], tf.int32)
	num_cols = tf.cast(example_features['width'], tf.int32)
	num_channels = tf.cast(example_features['depth'], tf.int32)
	# Reshape the image back to its original shape:
	tensor_image = tf.reshape(tensor_image, [num_rows, num_cols, num_channels])
	return tensor_image, tensor_label

### Apply your augmentations to individual images and labels here:
def augment(tensor_image, tensor_label, num_classes):
	# Normalize image to have values in [0,1]:
	tensor_image = normalize_tensor(tensor_image)
	# Transform label into its one-hot representation:
	tensor_label = tf.one_hot(tensor_label, num_classes)
	return tensor_image, tensor_label

### Decode and augment data:
def decode_and_augment(ciphered_example, num_classes):
	# Decode example:
	tensor_image, tensor_label = decode_example(ciphered_example)
	# Apply augmentations:
	tensor_image, tensor_label = augment(tensor_image, tensor_label, num_classes)
	return tensor_image, tensor_label

### Set up a generator that processes TFrecords input files with multi-threading enabled:
def GeneratorFromTFrecords(filenames, batch_size, num_classes, seed=None):
	num_shards = len(filenames)
	# Load and shuffle shards' filenames:
	filenames_dataset = tf.data.Dataset.list_files(filenames, shuffle=True, seed=seed)
	# Load asynchronously and in parallel on the CPU the records from different TFrecords shards:
	dataset = filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
										   cycle_length=num_shards,
										   num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# Shuffle TFrecords dataset (local randomness) then repeat dataset as much as needed (set a rather small buffer size because TFrecords were pre-shuffled):
	dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000, count=None))        
	# Decode + augment asynchronously and in parallel on the CPU the records while generating batches at the same time:
	dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda x: decode_and_augment(x, num_classes),
															   batch_size,
															   drop_remainder=True,
															   num_parallel_calls=tf.data.experimental.AUTOTUNE))
	# Prefetch asynchronously and in parallel on the CPU the next batches:
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
	# Create an iterator to run through the whole dataset once:
	dataset_iterator = dataset.make_one_shot_iterator()
	# Fetch the next batch and corresponding labels:
	next_batch_images, next_batch_labels = dataset_iterator.get_next()
	next_batch_images_shape = tensor2nparray(tf.shape(next_batch_images))
	# If necessary, resize images using bilinear interpolation:
	if not next_batch_images_shape[1] == 32 or not next_batch_images_shape[2] == 32:
		next_batch_images = tf.image.resize_bilinear(next_batch_images,
													 size=[32, 32],
													 align_corners=True)
	# If necessary, duplicate along 4th axis to recover RGB colors as concatenated copies of original grayscale (do it here because of the if, which requires actual data to flow in):
	if next_batch_images_shape[3] == 1:
		next_batch_images = tf.manip.tile(next_batch_images, [1, 1, 1, 3])
	# Don't forget to edit the static shape of the tensor (otherwise it will be [None, None, None, None] which will cause errors...)
	next_batch_images_shape = tensor2nparray(tf.shape(next_batch_images))
	next_batch_images.set_shape(next_batch_images_shape)
	return next_batch_images, next_batch_labels

### Minimal architecture:
def load_simple_model_tensorflow(input_tensor, num_classes):
	output_tensor = keras.layers.Flatten(input_shape=(32, 32, 3))(input_tensor)
	output_tensor = keras.layers.Dense(num_classes, activation='softmax')(output_tensor) 
	simple_model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
	return simple_model

### Initialize simple model:
def initialize_simple_model(batchs_images_tensor, batchs_labels_tensor, num_classes, 
						    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
						    summary=False):
	# Load simple model:
	model_input_tensor = keras.layers.Input(tensor=batchs_images_tensor)
	simple_model = load_simple_model_tensorflow(model_input_tensor, num_classes)
	simple_model.compile(optimizer=optimizer,
						 loss='categorical_crossentropy',
						 metrics=['accuracy'],
						 target_tensors=[batchs_labels_tensor])
	if summary:
		simple_model.summary()
	return simple_model

### Test TFrecords generator on a simple architecture:
def main(argv):
	try:                                
		opts, args = getopt.getopt(argv, "b:e:h", ["batch_size=", "epochs=", "help"])
	except getopt.GetoptError as error:  
		print(str(error))        
		usage()                         
		sys.exit(2) 
	## Default parameter values:    
	num_epochs = 5   
	batch_size = 64
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
		else:      
			print("Unhandled option !")    
			usage()                         
			sys.exit(2)                             
	if not len(args) == 1:
		print("Unique argument specifying path to a file with paths to TFrecords, path to the classes dictionary file and path to data information file is required !")
		usage()
		sys.exit(2)
	paths2parsable_files = args[0]
	# Fix randomness:
	random.seed = 951
	# Parse dataset informations, classes dictionary and shards' filenames of the dataset:
	num_train_samples, num_val_samples, _, num_classes, classes_dict, train_filenames, val_filenames, test_filenames = general_parser(paths2parsable_files)
	# Number of steps per epoch:
	steps_per_epoch = int(np.floor(num_train_samples / float(batch_size)))
	# Validation steps:
	validation_steps = int(np.floor(num_val_samples / float(batch_size)))
	# Set up batch generators:
	train_batch_images, train_batch_labels = GeneratorFromTFrecords(train_filenames, batch_size, num_classes, seed=random.seed)
	val_batch_images, val_batch_labels = GeneratorFromTFrecords(val_filenames, batch_size, num_classes, seed=random.seed)
	# Open a session to perform training:
	K.get_session()
	# Define a simple training model with input tensors:
	train_simple_model = initialize_simple_model(train_batch_images, train_batch_labels, num_classes, summary=True)
	# Define a validation model, copied from the training model: (with input tensors validation is not supported natively, need to use a custom callback with validation model as argument):
	val_simple_model = initialize_simple_model(val_batch_images, val_batch_labels, num_classes)
	# Train and validate after each epoch steps with custom validation callback for input tensors:
	train_simple_model.fit(epochs=num_epochs,
						   steps_per_epoch=steps_per_epoch,
						   callbacks=[EvaluateInputTensor(val_model=val_simple_model, steps=validation_steps)])
	# Clean the current session:
	K.clear_session()

### Passing command line to the main():	
if __name__== "__main__":
	verbose = False
	main(sys.argv[1:])
