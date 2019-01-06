#!/bin/bash

export PYTHONPATH=`pwd`

python model/label_transform/hierarchical_label_transformer.py \
model/data_download/preprocessing/output/data_with_hierarchical_label.csv \
model/label_transform/output/

