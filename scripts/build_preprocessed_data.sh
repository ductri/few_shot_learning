#!/bin/bash

export PYTHONPATH=`pwd`

python model/data_download/preprocessing/preprocess.py \
model/data_download/preprocessing/output/data_with_hierarchical_label.csv \
model/data_download/preprocessing/output/preprocessed_data_with_hierarchical_label.csv
