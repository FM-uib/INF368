#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script requires exactly one argument: the dataset path to process."
	exit 1
fi
dir_name=$1
new_dir="split_resized_dataset"
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
echo "Generating images paths files"
SECONDS=0
cd ../train
train_paths="../../train_images_paths"
if [ -f ${train_paths} ]; then
  rm ${train_paths}
fi
touch ${train_paths}
train_paths=../${train_paths}
for f in *; do
    	if [ -d ${f} ]; then
		cd ./${f}
		for g in $(ls *.jpg); do
			printf "./split_resized_dataset/train/${f}/${g}\n" >> ${train_paths}
		done
		cd ../
		fi
done
cd ../val
val_paths="../../val_images_paths"
if [ -f ${val_paths} ]; then
  rm ${val_paths}
fi
touch ${val_paths}
val_paths=../${val_paths}
for f in *; do
    	if [ -d ${f} ]; then
		cd ./${f}
		for g in $(ls *.jpg); do
			printf "./split_resized_dataset/val/${f}/${g}\n" >> ${val_paths}
		done
		cd ../
		fi
done
cd ../test
test_paths="../../test_images_paths"
if [ -f ${test_paths} ]; then
  rm ${test_paths}
fi
touch ${test_paths}
test_paths=../${test_paths}
for f in *; do
    	if [ -d ${f} ]; then
		cd ./${f}
		for g in $(ls *.jpg); do
			printf "./split_resized_dataset/test/${f}/${g}\n" >> ${test_paths}
		done
		cd ../
		fi
done
cd ../../
all_paths="./paths_to_parsable_files"
if [ -f ${all_paths} ]; then
  rm ${all_paths}
fi
touch ${all_paths}
printf "info_dataset: ./info_dataset\n" >> ${all_paths}
printf "dictionary: ./classes_dict\n" >> ${all_paths}
printf "train: ./train_images_paths\n" >> ${all_paths}
printf "val: ./val_images_paths\n" >> ${all_paths}
printf "test: ./test_images_paths\n" >> ${all_paths}
echo "This took:"
date +%T -d "1/1 + $SECONDS sec"
echo "--------------------------------------------------------------------------------"
echo "Procedure complete!"
echo ""


