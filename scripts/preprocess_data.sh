#!/bin/bash

export PYTHONPATH=`pwd`

python preprocess/preprocessor.py \
data_download/output/tulanh_filter_100.csv \
preprocess/output/preprocessed_data.csv \
100
