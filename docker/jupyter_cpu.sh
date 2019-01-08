#!/bin/bash

sudo docker logs -f --timestamps $(sudo docker run -d -e PYTHONIOENCODING=utf-8 --rm --name=few_shot_jupyter_cpu \
-v `pwd`:/source \
-v `pwd`/../dataset:/dataset:ro \
-p 1812:1812 \
trind/full-item-cpu /bin/bash -c "jupyter notebook --port=1812 --allow-root")
