#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : finetune the autoencoder for image restoration with additional skip connection. 
# Model : AutoUNet

path_config=$1
id_gpu=${2:-0}

if [[ -z $path_config ]]; then
    echo -e "\033[31mError: no config \033[0m" >&2
    echo -e "\033[31mError: hint: configs/finetune/autounet_bf.yaml \033[0m" >&2
    exit 1
fi

echo "config: ${path_config}"
echo "id_gpu: ${id_gpu}"


python main.py --base $path_config -t --gpus ${id_gpu},
