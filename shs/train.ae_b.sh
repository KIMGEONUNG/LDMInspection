#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : Train decoder using pre-acquired latent features, resulting from diffusion process
# Model : ldm.models.fuser.FusionKL8Feat

python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_fusion_b.yaml -t --gpus 0,
