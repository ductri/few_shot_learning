#!/bin/bash

docker logs -f --timestamps $(nvidia-docker run -d -e PYTHONIOENCODING=utf-8 --rm --name=intent_jupyter \
-v `pwd`:/source \
-p 1811:1811 \
trind/full-item-gpu /bin/bash -c "jupyter notebook --port=1811 --allow-root")
