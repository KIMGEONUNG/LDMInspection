#!/usr/bin/env python

import argparse
from ldm.util import instantiate_from_config
import torch
import os
import torch, torchvision
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor, Resize
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from pycomar.images import show_img, show3plt


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def extract_name(path: str):
    return path.split('/')[-1].split('.')[0]


class ReconDataset(Dataset):

    def __init__(self, root="inputs/openimage", dev="cuda:0"):
        self.root = root
        self.samples = sorted(glob(os.path.join(root, "*")))
        self.totensor = ToTensor()
        self.dev = dev

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path_img = self.samples[index]
        name = extract_name(path_img)
        img = self.loader_img(path_img)
        x = self.totensor(img)[None, ...]
        x = x * 2 - 1

        return x.to(self.dev), img, name

    def loader_img(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--path_input_dir', default='inputs/openimage')
    p.add_argument('-o',
                   '--path_output_dir',
                   default='outputs/recon_openimageB')
    p.add_argument('--one_file', action='store_true')
    p.add_argument('--error_map', action='store_true')
    p.add_argument(
        '--targets',
        nargs='+',
        type=str,
        default=[
            # "vq-f4",
            # "vq-f4-noattn",
            # "vq-f8",
            # "vq-f8-n256",
            # "vq-f16",
            "kl-f8",
            # "kl-f32",
            # "kl-f16",
            # "kl-f4",
        ],
    )
    return p.parse_args()


def num_param(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def main():
    args = parse()
    os.makedirs(args.path_output_dir, exist_ok=True)

    dataset = ReconDataset(args.path_input_dir)
    for target in args.targets:
        path_config = "models/first_stage_models/{}/config.yaml".format(target)
        path_model = "models/first_stage_models/{}/model.ckpt".format(target)
        config = OmegaConf.load(path_config)
        model = load_model_from_config(config, path_model)
        print("Num parameter of model  : ", num_param(model))
        print("Num parameter of encoder: ", num_param(model.encoder))
        print("Num parameter of decoder: ", num_param(model.decoder))
        for x, img, name in dataset:
            x_hat, _ = model(x, sample_posterior=True)

            # CORRECT IMAGE SIZES
            x, x_hat = x[0].cpu(), x_hat[0].cpu()
            x = Resize(x_hat.shape[-2:])(x)

            diff = torch.tensor([])
            if args.error_map:
                diff = (x - x_hat).abs().div(2).mean(-3).mul(255).to(torch.int)
                colormap_name = 'jet'
                cmap = plt.get_cmap(colormap_name)
                diff = cmap(diff)[..., :3]
                diff = diff * 2 - 1
                diff = torch.Tensor(diff).permute(2, 0, 1)

            if args.one_file:
                output = torch.cat([x, x_hat, diff], dim=-1)
                output = output.add(1).div(2).clamp(0, 1)
                output = ToPILImage()(output)
                path_hat = os.path.join(args.path_output_dir,
                                        "{}_{}.jpg".format(name, target))
                output.save(path_hat, quality=100, subsampling=0)
            else:
                x_hat = x_hat.add(1).div(2).clamp(0, 1)
                x_hat_img = ToPILImage()(x_hat)
                path_orig = os.path.join(args.path_output_dir,
                                         "{}.jpg".format(name))
                path_hat = os.path.join(args.path_output_dir,
                                        "{}_{}.jpg".format(name, target))
                img.save(path_orig, quality=100, subsampling=0)
                x_hat_img.save(path_hat, quality=100, subsampling=0)
        del model


if __name__ == "__main__":
    main()
