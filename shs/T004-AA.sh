#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

python main.py --base configs/finetune/T004-AA.yaml -t --gpus 0,
