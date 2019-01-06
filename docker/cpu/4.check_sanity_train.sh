#!/bin/bash

docker logs -f --timestamps $(docker run -d -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-cpu /bin/bash -c "/source/scripts/4.check_sanity_train.sh")
