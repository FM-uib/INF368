#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script requires exactly one argument: the dataset path to process."
	exit 1
fi
dir_name=$1
new_dir="split_resized_TFrecords_dataset"
if [ -d ${new_dir} ]; then
  rm -rf ./${new_dir}
fi
mkdir ./${new_dir}
SECONDS=0
cd ${dir_name}
classes_dict=../classes_dict
if [ -f ${classes_dict} ]; then
  rm ${classes_dict}
fi
touch ${classes_dict}
echo ""
square_size=224 # resize value set to 224 for ResNet compatibility
cmpt=0
for f in *; do
    	if [ -d ${f} ]; then
		printf "${f} : ${cmpt}\n" >> ${classes_dict}
		((++cmpt))
		python3 ../split_and_resize.py --data_dir ./${f} --output_dir ../${new_dir} --category ${f} --square_size ${square_size}
    	fi
done
cd ../${new_dir}/train
info_dataset=../../info_dataset
if [ -f ${info_dataset} ]; then
  rm ${info_dataset}
fi
touch ${info_dataset}
num_train_samples=0
for f in *; do
		if [ -d ${f} ]; then
		cd ./${f}
		((num_train_samples+=$(ls -Uba1 | grep ^train | wc -l)))
		cd ../
		fi
done
cd ../val
num_val_samples=0
for f in *; do
		if [ -d ${f} ]; then
		cd ./${f}
		((num_val_samples+=$(ls -Uba1 | grep ^val | wc -l)))
		cd ../
		fi
done
cd ../test
num_test_samples=0
for f in *; do
		if [ -d ${f} ]; then
		cd ./${f}
		((num_test_samples+=$(ls -Uba1 | grep ^test | wc -l)))
		cd ../
		fi
done
printf "num_train_samples: ${num_train_samples}\n" >> ${info_dataset}
printf "num_val_samples: ${num_val_samples}\n" >> ${info_dataset}
printf "num_test_samples: ${num_test_samples}\n" >> ${info_dataset}
printf "num_classes: ${cmpt}\n" >> ${info_dataset}
echo "Splitting dataset and resizing images took:" 
date +%T -d "1/1 + $SECONDS sec"
echo "--------------------------------------------------------------------------------"
SECONDS=0
cd ../
# num_shards chosen accordingly to a train/val/test split 0.8/0.1/0.1
num_shards_train=40 # number of TFrecords produced for train
num_shards_val=5 # number of TFrecords produced for val
num_shards_test=5 # number of TFrecords produced for test
python3 ../fuse_and_conv2TFrecords.py --data_dir ./ --dictionary ${classes_dict} --output_dir ./train --shards ${num_shards_train} --type "train"
python3 ../fuse_and_conv2TFrecords.py --data_dir ./ --dictionary ${classes_dict} --output_dir ./val --shards ${num_shards_val} --type "val"
python3 ../fuse_and_conv2TFrecords.py --data_dir ./ --dictionary ${classes_dict} --output_dir ./test --shards ${num_shards_test} --type "test"
echo "Fusing categories and converting shards to TFrecords took:" 
date +%T -d "1/1 + $SECONDS sec"
echo "--------------------------------------------------------------------------------"
echo "Cleaning directories and generating TFrecords paths files"
SECONDS=0
cd ./train
rm -rf */
train_paths="../../train_TFrecords_paths"
if [ -f ${train_paths} ]; then
  rm ${train_paths}
fi
touch ${train_paths}
for f in $(ls *.tfrecords); do
	printf "./split_resized_TFrecords_dataset/train/${f}\n" >> ${train_paths}
done
cd ../val
rm -rf */
val_paths="../../val_TFrecords_paths"
if [ -f ${val_paths} ]; then
  rm ${val_paths}
fi
touch ${val_paths}
for f in $(ls *.tfrecords); do
	printf "./split_resized_TFrecords_dataset/val/${f}\n" >> ${val_paths}
done
cd ../test
rm -rf */
test_paths="../../test_TFrecords_paths"
if [ -f ${test_paths} ]; then
  rm ${test_paths}
fi
touch ${test_paths}
for f in $(ls *.tfrecords); do
	printf "./split_resized_TFrecords_dataset/test/${f}\n" >> ${test_paths}
done
cd ../../
all_paths="./paths_to_parsable_files"
if [ -f ${all_paths} ]; then
  rm ${all_paths}
fi
touch ${all_paths}
printf "info_dataset: ./info_dataset\n" >> ${all_paths}
printf "dictionary: ./classes_dict\n" >> ${all_paths}
printf "train: ./train_TFrecords_paths\n" >> ${all_paths}
printf "val: ./val_TFrecords_paths\n" >> ${all_paths}
printf "test: ./test_TFrecords_paths\n" >> ${all_paths}
echo "This took:"
date +%T -d "1/1 + $SECONDS sec"
echo "--------------------------------------------------------------------------------"
echo "Procedure complete!"
echo ""


