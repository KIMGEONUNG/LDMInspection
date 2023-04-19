#!/bin/bash

set -e
source config.sh
source $condapath
conda activate ldm

# Goal  : Train decoder using pseudo GT latent features, resulting from E_{psi}(I_{GT})
# Model : ldm.models.fuser.FusionKL8

python main.py --base configs/autoencoder/autoencoder_kl_32x32x4_fusion.yaml -t --gpus 0,
