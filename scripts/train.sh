#!/bin/bash

export PYTHONPATH=`pwd`

python train/run_trainer.py \
SNN1 \
data_for_train/output/ \
--vocab=we/output/vocab.txt \
--batch_size=64 \
--num_epochs=20 \
--eval_interval=100 \
--word_embedding_npy=we/output/we_weights.npy

