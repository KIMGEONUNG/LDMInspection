#!/bin/bash

conda env create -f environment.yaml
conda activate ldm
# pip install packaging==21.3
# pip install 'torchmetrics<0.8'
# pip install kornia==0.5
# pip install academictorrents # confict occurs, but work
