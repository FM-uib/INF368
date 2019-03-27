#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script requires exactly one argument: the dataset path to process."
	exit 1
fi
dir_name=$1
new_dir="split_resized_images"
if [ -d ${new_dir} ]; then
  rm -rf ./${new_dir}
fi
mkdir ./${new_dir}
SECONDS=0
cd ${dir_name}
echo ""
square_size=224 # resize value set to 224 for ResNet compatibility
cmpt=0
for f in *; do
    	if [ -d ${f} ]; then
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
echo "Procedure complete!"
echo ""


