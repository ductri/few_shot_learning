#!/bin/bash

export PYTHONPATH=`pwd`

tensorboard --logdir=model/training/output/summary --port=2910
