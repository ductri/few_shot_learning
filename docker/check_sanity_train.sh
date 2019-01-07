#!/bin/bash

docker logs -f --timestamps $(nvidia-docker run -d -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-gpu /bin/bash -c "/source/scripts/check_sanity_train.sh")
