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
from torchvision.transforms import ToPILImage, ToTensor
from glob import glob
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('-p',
                   '--path_log',
                   default='logs/2023-04-19T19-04-27_autounet_bf/')
    p.add_argument('-n', '--num_sample', type=str, default=100)
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
    dir_output = join(args.path_log, "inference")
    os.makedirs(dir_output, exist_ok=True)

    path_metric = join(dir_output, "metric.yaml")
    try:
        os.remove(path_metric)
    except OSError:
        pass

    # FIND TARGET FILES
    name_model = sorted(os.listdir(dir_model))[-1]
    # if "last" in name_model:
    #     name_model = sorted(os.listdir(dir_model))[-2]
    name_config = [p for p in os.listdir(dir_config) if "project" in p][0]
    path_config = join(dir_config, name_config)
    path_model = join(dir_model, name_model)
    print("MODEL  :", path_model)
    print("CONFIG :", path_config)

    # DEFINE INPUT TRANSFORMATION
    def noise(x):
        sigma = max(0.2, random.random() / 2)
        x = ToTensor()(x)
        n = torch.randn_like(x)
        n.normal_(mean=0, std=sigma)
        x = x + n
        x = x.clamp(0, 1)
        x = ToPILImage()(x)
        return x

    def bf(x):
        x = np.asarray(x)
        filter_d = 10
        filter_sigmaColor = 50
        filter_sigmaSpace = 50

        x_hat = cv2.bilateralFilter(
            x,
            d=filter_d,
            sigmaColor=filter_sigmaColor,
            sigmaSpace=filter_sigmaSpace,
        )
        x_hat = Image.fromarray(x_hat)
        return x_hat

    if "bf" in args.path_log:
        fn = bf
    elif "noise" in args.path_log:
        fn = noise
    elif "blur" in args.path_log:
        # fn = GaussianBlur(5, (0.1, 3.0))
        # fn = GaussianBlur(5, (3.1, 4.0))
        # fn = GaussianBlur(5, (4.0, 6.0))
        fn = GaussianBlur(5, (6.0, 8.0))
    else:
        raise

    # LOAD CONFIG and model
    config = OmegaConf.load(path_config)
    model = load_model_from_config(config, path_model)

    # INFERENCE LOOP
    # with model.ema_scope():
    for i, path in enumerate(
            sorted(
                glob("DATASET/test/transform_test/*.jpg"))[:args.num_sample]):

        img_gt = Image.open(path).convert('RGB')
        img_input = fn(img_gt)

        x_input = ToTensor()(img_input) * 2 - 1
        x_input = x_input[None, ...].cuda()

        recon, posterior, intermids = model(x_input, sample_posterior=False)
        recon = ((recon + 1) / 2).clamp(0, 1)
        recon = ToPILImage()(recon.cpu()[0])

        grid = merge_pil(img_input, recon, img_gt)
        grid.save(join(dir_output, f"{i:04}.jpg"), quality=100, sunbsampling=0)

        psnr = PSNR(img_gt, recon)

        with open(path_metric, "a") as myfile:
            myfile.write(f"{i:04}: {psnr}\n")


if __name__ == "__main__":
    main()
