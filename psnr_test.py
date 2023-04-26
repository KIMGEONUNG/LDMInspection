#!/usr/bin/env python

import argparse
import cv2
import random
from ldm.util import instantiate_from_config
import torch
import os
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor, Resize, CenterCrop, Compose
from glob import glob
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm



def parse():
    p = argparse.ArgumentParser()
    p.add_argument('-p',
                   '--path_log',
                   default='logs/2023-04-20T10-46-41_autounet_mult')
    p.add_argument('-n', '--num_sample', type=int, default=100)
    return p.parse_args()


def PSNR(im1, im2):
    x1 = ToTensor()(im1).permute(2, 1, 0).numpy()
    x2 = ToTensor()(im2).permute(2, 1, 0).numpy()

    return compare_psnr(x1, x2)


@torch.no_grad()
def main():
    args = parse()

    # DEFINE DIRECTORY PATH
    dir_model = join(args.path_log, "checkpoints")
    dir_config = join(args.path_log, "configs")
    dir_output = join(args.path_log, "gen_dataset")

    t = Compose([
        Resize(512),
        CenterCrop(512),
    ])

    targets = sorted(glob(f"{dir_output}/*.jpg"))
    if args.num_sample > 0:
        targets = targets[:args.num_sample]

    psnrs = []
    paths = []
    for i, path in enumerate(tqdm(targets)):

        img_de = Image.open(path).convert('RGB')
        img_gt = Image.open(path.replace(
            dir_output, "DATASET/openimage/train")).convert('RGB')
        img_gt = t(img_gt)
        psnr = PSNR(img_gt, img_de)
        psnrs.append(psnr)
        paths.append(path)

    avg = sum(psnrs) / len(psnrs)

    with open(join(args.path_log, 'psnrs.txt'), 'w') as f:
        f.writelines([f"{str(m)},{p.split('/')[-1]}\n" for m, p in zip(psnrs, paths)])
    with open(join(args.path_log, 'psnrs_avg.txt'), 'w') as f:
        f.write(str(avg))


if __name__ == "__main__":
    main()
