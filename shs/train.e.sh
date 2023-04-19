#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : train shortcut network
# Model : AutoencoderKL_E

python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_e.yaml -t --gpus 0,
