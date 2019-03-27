"""Split a dataset into train/val/test and resize images to 224x224.
Input dataset must be orginized as follows:
	input_dataset/
		im_0.jpg
		im_1.jpg
		...
Images are also resized to match default input size of ResNet: 224x224
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

### Resize the image contained in `filename` and save it to the `output_dir`:
def resize_and_save(filename, output_dir, output_file_name, square_size):
	with Image.open(filename) as image:
		# Use bilinear interpolation instead of the default "nearest neighbor" method
		image = image.resize((square_size, square_size), Image.BILINEAR)
		image.save(os.path.join(output_dir, output_file_name).replace("\\","/"))

def main(args):
	data_dir = args.data_dir
	assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)
	filenames = os.listdir(data_dir)
	filenames = [os.path.join(data_dir, filename).replace("\\","/") for filename in filenames if filename.endswith('.jpg')]
	# Split the images in the dataset into 80% train, 10% val and 10% test
	# Make sure to always shuffle with a fixed seed so that the split is reproducible
	filenames.sort()
	random.seed(230)
	random.shuffle(filenames)
	split_1 = int(0.8 * len(filenames))
	split_2 = int(0.9 * len(filenames))
	train_filenames = filenames[:split_1]
	val_filenames = filenames[split_1:split_2]
	test_filenames = filenames[split_2:]
	filenames = {'train': train_filenames,
				 'val': val_filenames,
				 'test': test_filenames}
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	category_name = args.category
	square_size = int(args.square_size)
	# Preprocess train, val and test
	for split in ['train', 'val', 'test']:
		output_dir_split = os.path.join(output_dir, split).replace("\\","/")
		if not os.path.exists(output_dir_split):
			os.mkdir(output_dir_split)
		output_dir_split = os.path.join(output_dir_split, category_name).replace("\\","/")
		if not os.path.exists(output_dir_split):
			os.mkdir(output_dir_split)
		#print("--------------------------------------------------------------------------------")
		print("Splitting and resizing {} dataset {}, saving to {}".format(split, category_name, output_dir_split))
		cmpt = 0
		for filename in tqdm(filenames[split]):
			cmpt += 1
			output_file_name = split + "_" + category_name + "_" + str(cmpt) + '.jpg'
			resize_and_save(filename, output_dir_split, output_file_name, square_size)
		print("Done splitting and resizing " + split + " dataset " + category_name)
		print("--------------------------------------------------------------------------------")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./')
	parser.add_argument('--output_dir', default='./split_resized_dataset')
	parser.add_argument('--category', default="unknown")
	parser.add_argument('--square_size', default=224)
	args = parser.parse_args()
	main(args)

	
