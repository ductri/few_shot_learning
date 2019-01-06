#!/bin/bash

export PYTHONPATH=`pwd`

python data_for_train/create_dataset.py \
preprocess/output/preprocessed_data.csv \
we/output/vocab.txt \
--npz_dir=data_for_train/output/ \
--max_length=100 \
--dataset_size=10000

