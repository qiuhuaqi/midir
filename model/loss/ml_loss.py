from collections import OrderedDict

from torch import nn as nn
from utils.image import roi_crop


class mlLoss(nn.Module):
    def __init__(self,
                 sim_loss_fn,
                 sim_loss_name,
                 reg_loss_fn,
                 reg_loss_name,
                 reg_weight=0.0,
                 ml_lvls=1,
                 ml_weights=None):
        """
        Loss function

        Args:
            sim_loss_name: (str) Name of the similarity loss
            reg_loss_name: (str) Name of the regularisation loss
            reg_weight: (float) Weight on the spatial regularisation term
            mi_loss_cfg: (dict) MI loss configurations
            lncc_window_size: (int/tuple/list) Size of the LNCC window
            ml_lvls: (int) Numbers of levels for multi-resolution
            ml_weights: (tuple)
        """
        super(mlLoss, self).__init__()

        self.sim_loss_fn = sim_loss_fn
        self.sim_loss_name = sim_loss_name
        self.reg_loss_fn = reg_loss_fn
        self.reg_loss_name = reg_loss_name
        self.reg_weight = reg_weight

        # configure multi-resolutions
        self.ml_lvls = ml_lvls
        if ml_weights is None:
            ml_weights = (1.,) * ml_lvls
        self.ml_weights = ml_weights

    def forward(self, tar, warped_src, dvf_pred, roi_mask=None):
        # TODO: tar and warped_src should be lists of length ml_lvl

        if roi_mask is not None:
            # TODO: check if this changes the original data (pointer depth)
            tar = roi_crop(tar, roi_mask, dim=self.hparams.data.dim)
            warped_src = roi_crop(warped_src, roi_mask, dim=self.hparams.data.dim)

        # compute loss at multi-level
        loss_val = 0
        sim_loss_val = 0
        reg_loss_val = 0

        losses = OrderedDict()
        for l in range(self.ml_lvls):
            tar_l = tar[l]
            warped_src_l = warped_src[l]

            sim_loss_val_l = self.sim_loss_fn(tar_l, warped_src_l)
            reg_loss_val_l = self.reg_loss_fn(dvf_pred[-l - 1]) * self.reg_weight

            loss_val_l = sim_loss_val_l + reg_loss_val_l

            sim_loss_val = sim_loss_val + sim_loss_val_l * self.ml_weights[l]
            reg_loss_val = reg_loss_val + reg_loss_val_l * self.ml_weights[l]
            loss_val = loss_val + loss_val_l * self.ml_weights[l]

            # record losses for all resolution levels
            if l > 0:
                losses_l = {f"loss_lv{l}": loss_val_l,
                            f"{self.sim_loss_name}_lv{l}": sim_loss_val_l,
                            f"{self.reg_loss_name}_lv{l}": reg_loss_val_l}
                losses.update(losses_l)

        # add overall loss
        losses.update({f"loss": loss_val,
                      f"{self.sim_loss_name}": sim_loss_val,
                      f"{self.reg_loss_name}": reg_loss_val})

        return losses
