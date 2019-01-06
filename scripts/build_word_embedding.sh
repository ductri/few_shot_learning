#!/bin/bash

export PYTHONPATH=`pwd`

python we/run_we.py \
preprocess/output/preprocessed_all_data.csv \
--vocab_file=we/output/vocab.txt \
--embedding_weights_file=we/output/we_weights
