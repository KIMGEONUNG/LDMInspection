from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import (Compose, Resize, CenterCrop, ToTensor,
                                    RandomCrop, ToPILImage)
import json
import torch
from os.path import join
import numpy as np


class OpenImagePairDataset(Dataset):

    def __init__(
        self,
        path_gt: str,
        path_in: str,
        path_idx: str,
        resize_res: int = 512,
        size_crop: int = 512,
    ):
        self.path_gt = path_gt
        self.path_in = path_in
        self.path_idx = path_idx

        self.resize_res = resize_res
        self.size_crop = size_crop

        self.sizing = Compose(
            [Resize(self.resize_res),
             RandomCrop(self.size_crop)])

        with open(self.path_idx) as f:
            self.seeds = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):
        name = self.seeds[i]
        path_gt = join(self.path_gt, name)
        path_in = join(self.path_in, name)

        img_in = Image.open(path_in).convert('RGB')  # I_re
        img_gt = Image.open(path_gt).convert('RGB')  # I_gt

        # MERGE & CROP & AND & SPLIT
        merge = torch.cat([ToTensor()(img_in), ToTensor()(img_gt)])
        img_in, img_gt = self.sizing(merge).chunk(2)
        img_in, img_gt = ToPILImage()(img_in), ToPILImage()(img_gt)

        img_in = np.array(img_in).astype(np.float32) / 255 * 2 - 1
        img_gt = np.array(img_gt).astype(np.float32) / 255 * 2 - 1

        return dict(image=img_gt, lf=img_in)


class OpenImageTripleDataset(Dataset):

    def __init__(
        self,
        path_gt: str,
        path_in: str,
        path_z: str,
        path_idx: str,
        resize_res: int = 512,
        size_crop: int = 512,
    ):
        self.path_gt = path_gt
        self.path_in = path_in
        self.path_z = path_z
        self.path_idx = path_idx

        self.resize_res = resize_res
        self.size_crop = size_crop

        self.sizing = Compose(
            [Resize(self.resize_res),
             RandomCrop(self.size_crop)])

        with open(self.path_idx) as f:
            self.seeds = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):
        name = self.seeds[i]
        path_gt = join(self.path_gt, f"{name}.jpg")
        path_in = join(self.path_in, f"{name}.jpg")
        path_z = join(self.path_z, f"{name}.pt")

        img_in = Image.open(path_in).convert('RGB')  # I_re
        img_gt = Image.open(path_gt).convert('RGB')  # I_gt
        z = Image.open(path_z)

        # MERGE & CROP & AND & SPLIT
        merge = torch.cat([ToTensor()(img_in), ToTensor()(img_gt)])
        img_in, img_gt = self.sizing(merge).chunk(2)
        img_in, img_gt = ToPILImage()(img_in), ToPILImage()(img_gt)

        img_in = np.array(img_in).astype(np.float32) / 255 * 2 - 1
        img_gt = np.array(img_gt).astype(np.float32) / 255 * 2 - 1

        return dict(image=img_gt, lf=img_in, z_hat=z)
