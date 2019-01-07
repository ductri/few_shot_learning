#!/bin/bash

export PYTHONPATH=`pwd`

python preprocess/preprocessor.py \
data_download/output/intents_3.csv \
preprocess/output/preprocessed_data.csv \
100
