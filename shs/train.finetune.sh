#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : finetune the autoencoder for image restoration with additional skip connection. 
# Model : AutoUNet

python main.py --base configs/finetune/autounet_bf.yaml -t --gpus 0,
