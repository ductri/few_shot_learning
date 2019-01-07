#!/bin/bash

export PYTHONPATH=`pwd`

python we/run_we.py \
preprocess/output/preprocessed_data.csv \
--vocab_file=we/output/intent/vocab.txt \
--embedding_weights_file=we/output/intent/we_weights
