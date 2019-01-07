#!/bin/bash

export PYTHONPATH=`pwd`

python train/run_trainer.py \
SNN2 \
data_for_train/output/omniglot \
--vocab=we/output/intent/vocab.txt \
--batch_size=32 \
--num_epochs=20 \
--eval_interval=100 \
--word_embedding_npy=we/output/intent/we_weights.npy \
--gpu_fraction=0.3


