#!/bin/bash

batch_size=64
num_epochs=2
python3 ./TFrecords_generator_simple_NN.py -b ${batch_size} -e ${num_epochs} ./paths_to_parsable_files
