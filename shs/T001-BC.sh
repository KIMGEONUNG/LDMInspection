#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

python main.py --base configs/finetune/T001-BC.yaml -t --gpus 0,
