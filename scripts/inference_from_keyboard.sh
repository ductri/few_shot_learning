#!/bin/bash

export PYTHONPATH=`pwd`

python server/inference.py \
model/training/output/saved_models/HierarchicalSimpleRNNModel/2018-10-31T05:07:47-400 \
model/vocab/output/vocab.txt \
model/label_transform/output/ \
100
