from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


def load_model_original():
    config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')
    config.model.params.ckpt_path = "models/first_stage_models/kl-f8/model.ckpt"
    model = instantiate_from_config(config.model)
    return model


def load_model_modified():
    config = OmegaConf.load(
        'configs/autoencoder/autoencoder_kl_32x32x4_fusion.yaml')
    config.model.params.ckpt_path = "logs/2023-03-29T14-41-19_autoencoder_kl_32x32x4_fusion/checkpoints/epoch=000000.ckpt"
    model = instantiate_from_config(config.model)
    return model


def load_dataset():
    config = OmegaConf.load(
        'configs/autoencoder/autoencoder_kl_32x32x4_fusion.yaml')
    config.data.params.train.params.size_crop = 512
    dataset = instantiate_from_config(config.data.params.train)
    return dataset


@torch.no_grad()
def main():

    model = load_model_original()
    model_hat = load_model_modified()
    dataset = load_dataset()

    for i, sample in enumerate(dataset):
        gt, lf = sample['image'], sample['lf']
        reshape = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None, ...]
        gt, lf = reshape(gt), reshape(lf)

        out_before, _= model(gt, False)
        out_after, _, _ = model_hat(gt, lf, False)

        out_after = (out_after - out_after.min()) / (out_after.max() - out_after.min())
        out_before = (out_before + 1) / 2
        lf = (lf + 1) / 2
        gt = (gt + 1) / 2
        grid = make_grid(torch.cat([lf, out_after, gt, out_before]))
        a = ToPILImage()(grid)
        a.save(f"output_baseline/{i}.jpg", quality=100, sunbsampling=0)
        print('hello world')


if __name__ == "__main__":
    main()
