#!/bin/bash

export PYTHONPATH=`pwd`

python model/label_transform/label_transformer.py \
model/preprocess/output/preprocessed_data.csv \
model/label_transform/output/tulanh_lb_transformer.pkl
