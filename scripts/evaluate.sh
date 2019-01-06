#!/bin/bash

export PYTHONPATH=`pwd`

python server/evaluate.py \
--path_to_params_dict=model/train/output/saved_models/CNN1/2019-01-04T15:26:59/tensor_name.pkl \
--path_to_model=model/train/output/saved_models/CNN1/2019-01-04T15:26:59-501 \
--path_to_lb_transformer=model/label_transform/output/tulanh_lb_transformer.pkl \
--path_to_lb_info=model/data_download/output/tulanh_standard_label.csv \
--path_to_data_npz_file=model/data_for_train/output/tulanh_80407/_eval.npz
