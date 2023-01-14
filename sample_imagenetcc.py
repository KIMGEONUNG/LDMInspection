#!/usr/bin/env python

from ldm.models.diffusion.ddim import DDIMSampler
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  #, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


model = get_model()
sampler = DDIMSampler(model)

classes = list(range(1000))
n_samples_per_class = 12

ddim_steps = 20
ddim_eta = 0.0
scale = 3.0  # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning({
            model.cond_stage_key:
            torch.tensor(n_samples_per_class * [1000]).to(model.device)
        })

        for class_label in classes:
            print(
                f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}."
            )
            xc = torch.tensor(n_samples_per_class * [class_label])
            c = model.get_learned_conditioning(
                {model.cond_stage_key: xc.to(model.device)})

            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=c,
                batch_size=n_samples_per_class,
                shape=[3, 64, 64],
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                         min=0.0,
                                         max=1.0)

            grid = make_grid(x_samples_ddim, nrow=4)
            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            path = "outputs/imagenet_cc/cc_%03d.jpg" % (class_label)
            img.save(path, quality=100, subsampling=0)
