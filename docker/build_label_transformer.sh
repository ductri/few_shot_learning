#!/bin/bash

sudo docker run -it -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-cpu /bin/bash -c "/source/scripts/build_label_transformer.sh"
