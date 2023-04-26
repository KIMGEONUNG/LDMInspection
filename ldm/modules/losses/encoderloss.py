import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import LPIPS


class EncoderLoss(nn.Module):

    def __init__(
        self,
        pixelloss_weight=1.0,
        perceptual_weight=0.5,
        feature_weight=1.0,
    ):

        super().__init__()
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        gt,
        recon,
        posterior_gt,
        posterior,
        split="train",
    ):

        rec_loss = torch.abs(gt.contiguous() - recon.contiguous()).mean()
        p_loss = self.perceptual_loss(gt.contiguous(),
                                      recon.contiguous()).mean()
        feat_loss = torch.abs(posterior_gt.mode().contiguous() -
                              posterior.mode().contiguous()).mean()

        loss = self.pixelloss_weight * rec_loss + self.perceptual_weight * p_loss + self.feature_weight * feat_loss

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/rec_loss".format(split): rec_loss.detach(),
            "{}/p_loss".format(split): p_loss.detach(),
            "{}/feat_loss".format(split): feat_loss.detach(),
        }
        return loss, log
