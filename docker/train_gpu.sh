#!/bin/bash

docker logs -f --timestamps $(docker run --runtime=nvidia -d -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-gpu /bin/bash -c "/source/scripts/train.sh")
