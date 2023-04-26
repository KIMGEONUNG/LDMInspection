#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : train shortcut network
# Model : AutoencoderKL_E

python main.py --base configs/finetune/shortcut_a.yaml -t --gpus 0,
