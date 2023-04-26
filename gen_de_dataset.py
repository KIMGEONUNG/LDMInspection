#!/usr/bin/env python

import argparse
from tqdm import tqdm
import cv2
import random
from ldm.util import instantiate_from_config
import torch
import os
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from glob import glob
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from multiprocessing import Pool


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('-p',
                   '--path_log',
                   default='logs/2023-04-20T10-46-41_autounet_mult')
    return p.parse_args()


def merge_pil(*imgs):
    cat = np.concatenate([np.array(img) for img in imgs], axis=1)
    cat = Image.fromarray(cat)
    return cat


def PSNR(im1, im2):
    x1 = ToTensor()(im1).permute(2, 1, 0).numpy()
    x2 = ToTensor()(im2).permute(2, 1, 0).numpy()

    return compare_psnr(x1, x2)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    model.cuda()
    model.eval()
    return model


def extract_name(path: str):
    return path.split('/')[-1].split('.')[0]


@torch.no_grad()
def main():
    args = parse()

    # DEFINE DIRECTORY PATH
    dir_model = join(args.path_log, "checkpoints")
    dir_config = join(args.path_log, "configs")
    dir_output = join(args.path_log, "gen_dataset")
    os.makedirs(dir_output, exist_ok=True)

    path_metric = join(dir_output, "metric.yaml")
    try:
        os.remove(path_metric)
    except OSError:
        pass

    # FIND TARGET FILES
    name_model = sorted(os.listdir(dir_model))[-1]
    name_config = [p for p in os.listdir(dir_config) if "project" in p][0]
    path_config = join(dir_config, name_config)
    path_model = join(dir_model, name_model)
    print("MODEL  :", path_model)
    print("CONFIG :", path_config)

    # LOAD CONFIG
    config = OmegaConf.load(path_config)

    # LOAD DATASET
    config.data.params.train.params.size_crop = 512
    config.data.params.train.params.splits = (1., 0., 0.)
    data = instantiate_from_config(config.data.params.train)
    dataLoader = DataLoader(data, batch_size=8)

    # LOAD MODEL
    model = load_model_from_config(config, path_model)

    for item in tqdm(dataLoader):
        de = item['lf'].permute(0, 3, 1, 2).cuda()
        paths = item['path']

        recon, posterior, intermids = model(de, sample_posterior=False)
        recon = ((recon + 1) / 2).clamp(0, 1)

        for path, rec in zip(paths, recon):
            img_rec = ToPILImage()(rec.cpu())
            path_out = join(dir_output, path.split('/')[-1])
            img_rec.save(path_out, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
