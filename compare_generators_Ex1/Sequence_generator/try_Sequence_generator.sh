#!/bin/bash

batch_size=64
num_epochs=2
num_workers=4
python3 ./Sequence_generator_simple_NN.py -b ${batch_size} -e ${num_epochs} -w ${num_workers} ./paths_to_parsable_files
