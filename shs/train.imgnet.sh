#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

python main.py --base configs/latent-diffusion/cin-ldm-vq-f8.yaml -t --gpus 0,
