#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm

CUDA_VISIBLE_DEVICES=0 python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_fusion_b.yaml -t --gpus 0,
