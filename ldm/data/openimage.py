from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop
import json
from os.path import join
from pathlib import Path
import math
import cv2
import numpy as np
from einops import rearrange
from typing import Any
import torch
import torchvision
from torchvision.transforms import ToTensor, ToPILImage, GaussianBlur
import random


class FusionOpenImageDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits=(0.9, 0.05, 0.05),
        resize_res: int = 512,
        size_crop: int = 512,
        target: str = "bilater",
        filter_d=10,
        filter_sigmaColor=50,
        filter_sigmaSpace=50,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.resize_res = resize_res
        self.size_crop = size_crop

        self.filter_d = filter_d
        self.filter_sigmaColor = filter_sigmaColor
        self.filter_sigmaSpace = filter_sigmaSpace

        self.sizing = Compose(
            [Resize(self.resize_res),
             RandomCrop(self.size_crop)])

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

        if target == "bilateral":
            self.fn=self.bilateral_filter
        elif target == "noise":
            self.fn=self.noise
        elif target == "blur":
            self.fn=GaussianBlur(7, (5.0, 7.0))
        else:
            raise

    def __len__(self) -> int:
        return len(self.seeds)

    def noise(self, x):
        sigma = random.random() / 2
        x = ToTensor()(x)
        n = torch.randn_like(x)
        n.normal_(mean=0, std=sigma)
        x = x + n
        x = x.clamp(0, 1)
        x = ToPILImage()(x)
        return x

    def downsample(self, x):
        raise

    def bilateral_filter(
        self,
        x,
    ):
        x = np.asarray(x)
        x_hat = cv2.bilateralFilter(
            x,
            d=self.filter_d,
            sigmaColor=self.filter_sigmaColor,
            sigmaSpace=self.filter_sigmaSpace,
        )
        x_hat = Image.fromarray(x_hat)
        return x_hat

    def __getitem__(self, i: int):
        path = self.seeds[i]
        path = join(self.path, path)

        img_gt = Image.open(path).convert('RGB')  # GT
        img_gt = self.sizing(img_gt)

        img_input = self.fn(img_gt)

        img_input = np.array(img_input).astype(np.float32) / 255 * 2 - 1
        img_gt = np.array(img_gt).astype(np.float32) / 255 * 2 - 1

        return dict(image=img_gt, lf=img_input)


class FusionOpenImageFeatDataset(Dataset):

    def __init__(
            self,
            path: str,
            split: str = "train",
            splits=(0.9, 0.05, 0.05),
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):
        name = self.seeds[i]
        path_lf = f"transform_lf/{name}.jpg"
        path_gt = f"transform/{name}.jpg"
        path_feat = f"restored_feat/{name}.pt"

        path_lf = join(self.path, path_lf)
        path_gt = join(self.path, path_gt)
        path_feat = join(self.path, path_feat)

        img_gt = Image.open(path_gt).convert('RGB')  # GT
        img_lf = Image.open(path_lf).convert('RGB')  # LF, actually bf
        feat = torch.load(path_feat)[0]

        img_gt = np.array(img_gt).astype(np.float32) / 255 * 2 - 1
        img_lf = np.array(img_lf).astype(np.float32) / 255 * 2 - 1

        return dict(image=img_gt, lf=img_lf, feat=feat)
