from collections import OrderedDict
from torch import nn as nn


class MultiResLoss(nn.Module):
    def __init__(self,
                 sim_loss_fn,
                 sim_loss_name,
                 reg_loss_fn,
                 reg_loss_name,
                 reg_weight=0.0,
                 ml_lvls=1,
                 ml_weights=None):
        """
        Multi-resolution Loss function

        Args:
            sim_loss_name: (str) Name of the similarity loss
            reg_loss_name: (str) Name of the regularisation loss
            reg_weight: (float) Weight on the spatial regularisation term
            mi_loss_cfg: (dict) MI loss configurations
            lncc_window_size: (int/tuple/list) Size of the LNCC window
            ml_lvls: (int) Numbers of levels for multi-resolution
            ml_weights: (tuple)
        """
        super(MultiResLoss, self).__init__()

        self.sim_loss_fn = sim_loss_fn
        self.sim_loss_name = sim_loss_name
        self.reg_loss_fn = reg_loss_fn
        self.reg_loss_name = reg_loss_name
        self.reg_weight = reg_weight

        # configure multi-resolutions and weighting
        self.ml_lvls = ml_lvls
        if ml_weights is None:
            ml_weights = (1.,) * ml_lvls
        self.ml_weights = ml_weights
        assert len(self.ml_weights) == self.ml_lvls

    def forward(self, tars, warped_srcs, flows):
        assert len(tars) == self.ml_lvls
        assert len(warped_srcs) == self.ml_lvls
        assert len(flows) == self.ml_lvls

        # compute loss at multi-level
        loss = 0
        sim_loss = 0
        reg_loss = 0

        losses = OrderedDict()
        for l, (tar, warped_src, flow, weight_l) in enumerate(zip(tars, warped_srcs, flows, self.ml_weights)):
            sim_loss_l = self.sim_loss_fn(tar, warped_src)
            reg_loss_l = self.reg_loss_fn(flow) * self.reg_weight
            loss_l = sim_loss_l + reg_loss_l

            sim_loss = sim_loss + sim_loss_l * weight_l
            reg_loss = reg_loss + reg_loss_l * weight_l
            loss = loss + loss_l * weight_l

            # record losses for all resolution levels
            losses_l = {f"loss_lv{l}": loss_l,
                        f"{self.sim_loss_name}_lv{l}": sim_loss_l,
                        f"{self.reg_loss_name}_lv{l}": reg_loss_l}
            losses.update(losses_l)

        # add overall loss
        losses.update({f"loss": loss,
                      f"{self.sim_loss_name}": sim_loss,
                      f"{self.reg_loss_name}": reg_loss})

        return losses
