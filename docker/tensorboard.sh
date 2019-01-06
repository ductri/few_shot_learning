#!/bin/bash

nvidia-docker run -d --rm -e PYTHONIOENCODING=utf-8 \
--name diacritics_tensorboard \
-v `pwd`:/source \
-p 2311:2311 \
trind/full-item-gpu /bin/bash -c "tensorboard --logdir=model/train/output/summary/ --port=2311"
