#!/bin/bash

docker run -ti --rm -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-cpu /bin/bash -c "/source/scripts/evaluate.sh"

