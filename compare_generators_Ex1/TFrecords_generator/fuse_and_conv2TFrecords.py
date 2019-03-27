"""Split a dataset into train/val/test and resize images to 224x224.
Input dataset must be orginized as follows:
	input_dataset/
		im_0.jpg
		im_1.jpg
		...
Images are also resized to match default input size of ResNet: 224x224
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
import sys
import numpy as np
import math
import tensorflow as tf

from PIL import Image
from tqdm import tqdm

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

### Classes correspondance:
def get_classes_dict(path_dictionary):
	classes_dict = {}
	with open(path_dictionary,'r') as dictionary:
		lines = dictionary.readlines()
		for line in lines:
			class_name, _, end_idx = get_first_word_in_string(line)
			class_value, _, _ = get_first_word_in_string(line[end_idx+2:])
			classes_dict[class_name] = int(class_value)
	return classes_dict

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

### Converts an image to tfrecords:
def convert_image(image, category_value, TFrecords_writer):
	rows = image.shape[0]
	cols = image.shape[1]
	depth = image.shape[2]
	image_raw = image.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(int(rows)),
																   'width': _int64_feature(int(cols)),
																   'depth': _int64_feature(int(depth)),
																   'label': _int64_feature(int(category_value)),
																   'image_raw': _bytes_feature(tf.compat.as_bytes(image_raw))}))
	TFrecords_writer.write(example.SerializeToString())

### Converts a shard to tfrecords:
def convert_shard(filenames, output_dir, dataset_type, shard_idx, num_shards, classes_dict):
	filenames = [filename for filename in filenames if filename.endswith('.jpg')]
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	TFrecords_filename = os.path.join(output_dir, dataset_type + "_shard_" + str(shard_idx) + '.tfrecords').replace("\\","/")
	print("Converting shard " + str(shard_idx) + "/" + str(num_shards) + " of " + dataset_type + " dataset to TFRecords")
	with tf.python_io.TFRecordWriter(TFrecords_filename) as TFrecords_writer:
		for filename in tqdm(filenames):
			with Image.open(filename) as image:
				np_image = np.array(image)
				if len(np_image.shape) == 2:
					np_image = np_image[:,:,np.newaxis]
				category = filename.split("/")[-1]
				category = category.split(".")[0]
				category = category.split("_")[1:-1]
				category = "_".join(category)
				category_value = classes_dict[category]
				convert_image(np_image, category_value, TFrecords_writer)
	print("Done converting shard " + str(shard_idx) + "/" + str(num_shards) + " of " + dataset_type + " dataset to TFRecords")

def main(args):
	data_dir = args.data_dir 
	output_dir = args.output_dir
	dataset_type = args.type
	all_categories = os.path.join(data_dir, dataset_type).replace("\\","/")
	assert os.path.isdir(all_categories), "Couldn't find the dataset at {}".format(data_dir)
	categories = os.listdir(all_categories)
	# Shuffle the order the categories are processed
	categories.sort()
	random.seed(421)
	random.shuffle(categories)
	filenames = []
	for category in categories:
		category_path = os.path.join(all_categories, category).replace("\\","/")
		category_filenames = os.listdir(category_path) 
		category_filenames = [os.path.join(category_path, filename).replace("\\","/") for filename in category_filenames]
		filenames += category_filenames
	# Shuffle the order images are processed
	random.seed(753)
	filenames.sort()
	random.shuffle(filenames)
	num_shards = int(args.shards)
	length_total = len(filenames)
	length_shard = int(math.ceil(length_total / float(num_shards)))
	print("Converting " + dataset_type + " dataset to TFRecords")
	path_dictionary = args.dictionary
	classes_dict = get_classes_dict(path_dictionary)
	for shard_idx in range(num_shards):
		begin_idx = shard_idx * length_shard
		end_idx = min((shard_idx + 1) * length_shard, length_total)
		filenames_current_shard = filenames[begin_idx:end_idx]
		convert_shard(filenames_current_shard, output_dir, dataset_type, (1 + shard_idx), num_shards, classes_dict)
	print("Done converting " + dataset_type + " dataset to TFRecords")
	print("--------------------------------------------------------------------------------")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./')
	parser.add_argument('--dictionary', default='./classes_dict')
	parser.add_argument('--output_dir', default='./fused_dataset')
	parser.add_argument('--shards', default=10)
	parser.add_argument('--type', default="")
	args = parser.parse_args()
	main(args)

	
