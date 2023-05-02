import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import LPIPS


class AutoUNetLoss(nn.Module):

    def __init__(
        self,
        w_shortcut=1.0,
        w_fusion=1.0,
        w_fusion_pix=1.0,
        w_fusion_lpips=1.0,
    ):

        super().__init__()
        self.lpips = LPIPS().eval()

        # WEIGHTS
        self.w_shortcut = w_shortcut

        self.w_fusion = w_fusion
        self.w_fusion_pix = w_fusion_pix
        self.w_fusion_lpips = w_fusion_lpips

    def forward(
        self,
        x_hat,
        x_hat_prime,
        gt,
        split="train",
    ):

        l_fusion_pix = (gt - x_hat).mean()
        l_fusion_lpips = self.lpips(gt, x_hat).mean()
        l_fusion = self.w_fusion_pix * l_fusion_pix + self.w_fusion_lpips * l_fusion_lpips

        l_shortcut = self.lpips(gt, x_hat_prime).mean()

        l_total = self.w_shortcut * l_shortcut + self.w_fusion * l_fusion

        log = {
            "{}/total_loss".format(split): l_total.clone().detach(),
            "{}/fusion_loss".format(split): l_fusion.detach(),
            "{}/shorcut_loss".format(split): l_shortcut.detach(),
        }
        return l_total, log
