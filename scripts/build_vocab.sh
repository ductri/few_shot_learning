#!/bin/bash

export PYTHONPATH=`pwd`

python model/vocab/vocab_builder.py model/data_download/preprocessing/output/data_with_hierarchical_label.csv \
model/vocab/output/vocab.txt \
--preprocessed_data_file=model/data_download/preprocessing/output/preprocessed_data_with_hierarchical_label.csv \
--vocab_size=10000

