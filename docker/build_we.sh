#!/bin/bash

sudo docker run -ti -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
trind/full-item-cpu /bin/bash -c "/source/scripts/build_word_embedding.sh"
