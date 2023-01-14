#!/usr/bin/env python

from ldm.util import instantiate_from_config
import torch
import os
import torch, torchvision
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from glob import glob
from PIL import Image
from pycomar.images import show_img, show3plt


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

    def __init__(self, root="inputs/recon", dev="cuda:0"):
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

@torch.no_grad()
def main():
    targets = [  # Types of autoencoder models
        "vq-f4",
        "vq-f4-noattn",
        "vq-f8",
        "vq-f8-n256",
        "vq-f16",
        "kl-f8",
        "kl-f32",
        "kl-f16",
        "kl-f4",
    ]
    path_output_dir = "outputs/recon"

    dataset = ReconDataset()
    for target in targets:
        path_config = "models/first_stage_models/{}/config.yaml".format(target)
        path_model = "models/first_stage_models/{}/model.ckpt".format(target)
        config = OmegaConf.load(path_config)
        model = load_model_from_config(config, path_model)
        for x, img, name in dataset:
            x_hat, _ = model(x)
            x_hat = x_hat.add(1).div(2).clamp(0, 1)[0]
            x_hat_img = ToPILImage()(x_hat)

            path_orig = os.path.join(path_output_dir, "{}.jpg".format(name))
            path_hat = os.path.join(path_output_dir,
                                    "{}_{}.jpg".format(name, target))
            img.save(path_orig, quality=100, subsampling=0)
            x_hat_img.save(path_hat, quality=100, subsampling=0)
        del model

if __name__ == "__main__":
    main()
