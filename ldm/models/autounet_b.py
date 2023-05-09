import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import (ResnetBlock, Upsample,
                                                Normalize, make_attn,
                                                nonlinearity, Downsample,
                                                Encoder, Decoder)
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AutoUNet_B(pl.LightningModule):
    """
    Handle three types
    - shortcut
    - fusion
    - joint
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        target,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        monitor=None,
    ):
        assert ddconfig["double_z"]
        assert target in ["shortcut", "fusion", "joint"]
        super().__init__()
        self.image_key = image_key
        self.target = target

        # LOAD MODEL
        self.shortcut = EncoderF(**ddconfig)
        self.fusion = DecoderF(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"],
                                          2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)

        # LOAD MODEL (FIX)
        self.encoder_fix = Encoder(**ddconfig)
        self.decoder_fix = Decoder(**ddconfig)
        self.quant_conv_fix = torch.nn.Conv2d(2 * ddconfig["z_channels"],
                                              2 * embed_dim, 1)
        self.post_quant_conv_fix = torch.nn.Conv2d(embed_dim,
                                                   ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

        # COPY WIEIGHT
        self.encoder_fix.load_state_dict(self.shortcut.state_dict(),
                                         strict=True)
        self.decoder_fix.load_state_dict(self.fusion.state_dict(),
                                         strict=False)
        self.quant_conv_fix.load_state_dict(self.quant_conv.state_dict(),
                                            strict=True)
        self.post_quant_conv_fix.load_state_dict(
            self.post_quant_conv.state_dict(), strict=True)
        print(f"Restored from {path}")

    def encode_fix(self, x):
        h = self.encoder_fix(x)
        moments = self.quant_conv_fix(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode_fix(self, z):
        z = self.post_quant_conv_fix(z)
        dec = self.decoder_fix(z)
        return dec

    def encode(self, x):
        h, intermids = self.shortcut(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, intermids

    def decode(self, z, intermids):
        z = self.post_quant_conv(z)
        dec = self.fusion(z, intermids)
        return dec

    def forward(self, x, sample_posterior=False):
        if self.target == "shortcut":
            posterior, _ = self.encode(x)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            dec = self.decode_fix(z)
            return dec, posterior
        elif self.target == "fusion":
            raise AssertionError
        elif self.target == "joint":
            raise AssertionError

        raise AssertionError

    def get_input(self, batch, k):
        if self.target == "shortcut":
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1,
                          2).to(memory_format=torch.contiguous_format).float()
            return x
        elif self.target == "fusion":
            raise AssertionError
        elif self.target == "joint":
            raise AssertionError
        raise AssertionError

    def training_step(self, batch, batch_idx, optimizer_idx):

        img_gt = self.get_input(batch, self.image_key)
        img_lf = self.get_input(batch, "lf")

        if self.target == "shortcut":
            reconstructions, posterior = self(img_lf)
        elif self.target == "fusion":
            raise AssertionError
        elif self.target == "joint":
            raise AssertionError

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(img_gt,
                                            reconstructions,
                                            posterior,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
            self.log("aeloss",
                     aeloss,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=True)
            self.log_dict(log_dict_ae,
                          prog_bar=False,
                          logger=True,
                          on_step=True,
                          on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                img_gt,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train")

            self.log("discloss",
                     discloss,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=True)
            self.log_dict(log_dict_disc,
                          prog_bar=False,
                          logger=True,
                          on_step=True,
                          on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):

        img_gt = self.get_input(batch, self.image_key)
        img_lf = self.get_input(batch, "lf")

        if self.target == "shortcut":
            reconstructions, posterior = self(img_lf)
        elif self.target == "fusion":
            raise AssertionError
        elif self.target == "joint":
            raise AssertionError

        aeloss, log_dict_ae = self.loss(img_gt,
                                        reconstructions,
                                        posterior,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val")

        discloss, log_dict_disc = self.loss(img_gt,
                                            reconstructions,
                                            posterior,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.target == "shortcut":
            opt = torch.optim.Adam(
                list(self.shortcut.parameters()) +
                list(self.quant_conv.parameters()),
                lr=lr,
                betas=(0.5, 0.9),
            )
        elif self.target == "fusion":
            opt = torch.optim.Adam(
                list(self.fusion.parameters()) +
                list(self.post_quant_conv.parameters()),
                lr=lr,
                betas=(0.5, 0.9),
            )
        elif self.target == "joint":
            opt = torch.optim.Adam(
                list(self.shortcut.parameters()) +
                list(self.quant_conv.parameters()) +
                list(self.fusion.parameters()) +
                list(self.post_quant_conv.parameters()),
                lr=lr,
                betas=(0.5, 0.9),
            )
        else:
            raise AssertionError

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        return [opt, opt_disc], []

    def get_last_layer(self):
        if self.target == "shortcut":
            return self.decoder_fix.conv_out.weight
        elif self.target == "fusion":
            return self.fusion.conv_out.weight
        elif self.target == "joint":
            return self.fusion.conv_out.weight
        raise AssertionError

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        gt = self.get_input(batch, self.image_key).to(self.device)
        lf = self.get_input(batch, "lf").to(self.device)

        x_hat, x_hat_prime, *_ = self(lf, gt)
        log["A_lf"] = lf
        log["B_gt"] = gt
        log["C_x_hat"] = x_hat
        log["C_x_hat_prime"] = x_hat_prime

        return log


class EncoderF(nn.Module):

    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 double_z=True,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 *
                                        z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        intermids = [hs[-1]]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            intermids.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, intermids


class DecoderF(nn.Module):

    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1, ) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # fusions
        self.fusion_1 = zero_module(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.fusion_2 = zero_module(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.fusion_3 = zero_module(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, intermids):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):

            if i_level == 1:
                h = h + self.fusion_3(intermids[2])
            if i_level == 0:
                h = h + self.fusion_2(intermids[1])

            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = h + self.fusion_1(intermids[0])

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
