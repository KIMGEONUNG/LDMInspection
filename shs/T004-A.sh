#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

python main.py --base configs/finetune/T004-A.yaml -t --gpus 0,
