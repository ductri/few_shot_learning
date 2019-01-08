#!/bin/bash

nvidia-docker run -d --rm -e PYTHONIOENCODING=utf-8 \
--name few_shot_tensorboard \
-v `pwd`:/source \
-p 1813:1813 \
trind/full-item-gpu /bin/bash -c "tensorboard --logdir=train/output/summary/ --port=1813"
