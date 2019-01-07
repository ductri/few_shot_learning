#!/bin/bash

export PYTHONPATH=`pwd`

python data_for_train/create_omniglot_dataset.py \
--data_dir=/dataset/omniglot/python/images_background/train \
--npz_dir=data_for_train/output/omniglot/ \
--dataset_size=50000 \
--name=eval
