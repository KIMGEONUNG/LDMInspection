#!/usr/bin/env python

import argparse
from ldm.util import instantiate_from_config
import torch
import os
from os.path import join
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from glob import glob
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from pathlib import Path
from tqdm import tqdm


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('-p',
                   '--path_log',
                   default='logs/2023-04-26T16-27-20_T001-BA/')
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
    cfn = Path(os.path.basename(__file__)).stem
    path_re = "DATASET/openimage_pair_a/test1K512x512_re"
    dir_model = join(args.path_log, "checkpoints")
    dir_config = join(args.path_log, "configs")

    dir_output = join("outputs", cfn)
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

    # LOAD CONFIG and model
    config = OmegaConf.load(path_config)
    model = load_model_from_config(config, path_model)

    # INFERENCE LOOP
    # with model.ema_scope():
    for i, path in enumerate(
            tqdm(sorted(glob(join(path_re, "*.jpg")))[:args.num_sample])):

        path_gt = path.replace('test1K512x512_re', 'test1K512x512')
        img_gt = Image.open(path_gt).convert('RGB')
        img_input = Image.open(path).convert('RGB')

        x_input = ToTensor()(img_input) * 2 - 1
        x_input = x_input[None, ...].cuda()

        recon, _= model(x_input)
        recon = ((recon + 1) / 2).clamp(0, 1)
        recon = ToPILImage()(recon.cpu()[0])

        grid = merge_pil(img_input, recon, img_gt)
        grid.save(join(dir_output, f"{i:04}.jpg"), quality=100, sunbsampling=0)

        psnr = PSNR(img_gt, recon)

        with open(path_metric, "a") as myfile:
            myfile.write(f"{i:04}: {psnr}\n")


if __name__ == "__main__":
    main()
