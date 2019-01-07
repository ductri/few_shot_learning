#!/bin/bash

sudo docker run -ti -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/source \
-v `pwd`/../dataset:/dataset:ro \
trind/full-item-cpu /bin/bash -c "/source/scripts/build_data_for_train.sh"
