from collections import OrderedDict

import torch
from torch import nn as nn
from torch.nn import functional as F


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Input shape (N, dim, *sizes), mode='foward', 'backward' or 'central'"""
    assert type(x) is torch.Tensor
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    if mode == "central":
        # TODO: implement central difference by 1d conv or dialated slicing
        raise NotImplementedError("Finite difference central difference mode")
    else:  # "forward" or "backward"
        # configure padding of this dimension
        paddings = [[0, 0] for _ in range(ndim)]
        if mode == "forward":
            # forward difference: pad after
            paddings[dim][1] = 1
        elif mode == "backward":
            # backward difference: pad before
            paddings[dim][0] = 1
        else:
            raise ValueError(f'Mode {mode} not recognised')

        # reverse and join sublists into a flat list (Pytorch uses last -> first dim order)
        paddings.reverse()
        paddings = [p for ppair in paddings for p in ppair]

        # pad data
        if boundary == "Neumann":
            # Neumann boundary condition
            x_pad = F.pad(x, paddings, mode='replicate')
        elif boundary == "Dirichlet":
            # Dirichlet boundary condition
            x_pad = F.pad(x, paddings, mode='constant')
        else:
            raise ValueError("Boundary condition not recognised.")

        # slice and subtract
        x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
                 - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

        return x_diff


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
        sim_loss = []
        reg_loss = []
        loss = []

        losses = OrderedDict()
        for lvl, (tar, warped_src, flow, weight_l) in enumerate(zip(tars, warped_srcs, flows, self.ml_weights)):
            sim_loss_l = self.sim_loss_fn(tar, warped_src) * weight_l
            reg_loss_l = self.reg_loss_fn(flow) * self.reg_weight * weight_l
            loss_l = sim_loss_l + reg_loss_l

            sim_loss.append(sim_loss_l)
            reg_loss.append(reg_loss_l)
            loss.append(loss_l)

            # record weighted losses for all resolution levels
            if self.ml_lvls > 1:
                losses.update(
                    {
                        f"{self.sim_loss_name}_lv{lvl}": sim_loss_l,
                        f"{self.reg_loss_name}_lv{lvl}": reg_loss_l,
                        f"loss_lv{lvl}": loss_l
                    }
                )

        # add overall loss to the dict
        losses.update(
            {
                f"{self.sim_loss_name}": torch.sum(torch.stack(sim_loss)),
                f"{self.reg_loss_name}": torch.sum(torch.stack(reg_loss)),
                f"loss": torch.sum(torch.stack(loss))
            }
        )
        return losses