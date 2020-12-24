from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


# Spatial derivatives by finite difference


def finite_diff(x, mode="central", boundary="Neumann"):
    """Input shape (N, dim, *sizes), mode='foward', 'backward' or 'central'"""
    if mode == "central":
        # TODO: this might be memory inefficient
        x_diff_forward = finite_diff_oneside(x, direction="forward", boundary=boundary)
        x_diff_backward = finite_diff_oneside(x, direction="backward", boundary=boundary)
        return [(x_diff_fw + x_diff_bw) / 2
                for (x_diff_fw, x_diff_bw) in zip(x_diff_forward, x_diff_backward)]
    else:
        # "forward" or "backward"
        return finite_diff_oneside(x, direction=mode, boundary=boundary)


def finite_diff_oneside(x, direction="forward", boundary="Neumann"):
    """
    Calculate one-sided finite difference, works with 2D/3D Pytorch Tensor or Numpy Array.
    Forward difference: dx[i] = x[i+1] - x[i]
    Backward difference: dx[i] = x[i] - x[i-1]

    Args:
        x: (torch.Tensor or numpy.ndarray, size/shape (N, dim, *size)) The array to differentiate.
        direction: (string) Direction of the finite difference appoximation, "forward" or "backward".
        boundary: (string) Boundary condition, "Neumann" or "Dirichlet".

    Returns:
        x_diff
    """

    ndim = x.ndim - 2
    sizes = x.shape[2:]

    # initialise finite difference list
    x_diff = []  # [x_dx, x_dy, x_dz]

    for i in range(ndim):
        # configure padding of this dimension
        # (don't use []*dim as it just replicates pointers of the one array)
        paddings = [[0, 0] for j in range(ndim)]

        if direction == "forward":
            # forward difference: pad after
            paddings[i][1] = 1
        else:
            # backward difference: pad before
            paddings[i][0] = 1


        if type(x) is np.ndarray:
            # add the first 2 dimensions for numpy
            paddings = [[0, 0], [0, 0]] + paddings

            # padding
            if boundary == "Neumann":
                # Neumann boundary condition
                x_pad = np.pad(x, paddings, mode='edge')
            elif boundary == "Dirichlet":
                # Dirichlet boundary condition
                x_pad = np.pad(x, paddings, mode='constant')
            else:
                raise ValueError("Boundary condition not recognised.")

            # slice and subtract
            x_diff += [x_pad.take(np.arange(1, sizes[i]+1), axis=i+2)
                       - x_pad.take(np.arange(0, sizes[i]), axis=i+2)]

        elif type(x) is torch.Tensor:
            # Pytorch uses last -> first dimension order
            paddings.reverse()
            # join sublists into a flat list
            paddings = [p for ppair in paddings for p in ppair]

            # padding
            if boundary == "Neumann":
                # Neumann boundary condition
                x_pad = F.pad(x, paddings, mode='replicate')
            elif boundary == "Dirichlet":
                # Dirichlet boundary condition
                x_pad = F.pad(x, paddings, mode='constant')
            else:
                raise ValueError("Boundary condition not recognised.")

            # slice and subtract
            x_diff += [x_pad.index_select(i+2, torch.arange(1, sizes[i]+1).to(device=x.device))
                       - x_pad.index_select(i+2, torch.arange(0, sizes[i]).to(device=x.device))]

        else:
            raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")

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