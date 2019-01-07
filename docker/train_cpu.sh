#!/bin/bash

sudo docker logs -f --timestamps $(sudo docker run -d -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-cpu /bin/bash -c "/source/scripts/train.sh")
