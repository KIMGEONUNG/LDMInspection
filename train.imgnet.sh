#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin-ldm-vq-f8.yaml -t --gpus 0,
