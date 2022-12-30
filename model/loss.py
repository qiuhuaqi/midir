import math
import torch
from torch import nn as nn
from torch.nn import functional as F


class LossFn(nn.Module):
    def __init__(
        self, sim_loss_fn, reg_loss_fn, sim_loss_weight=1.0, reg_loss_weight=1.0
    ):
        super(LossFn, self).__init__()
        self.sim_loss_fn = sim_loss_fn
        self.sim_loss_weight = sim_loss_weight
        self.reg_loss_fn = reg_loss_fn
        self.reg_loss_weight = reg_loss_weight

    def forward(self, tar, warped_src, u):
        sim_loss = self.sim_loss_fn(tar, warped_src)
        reg_loss = self.reg_loss_fn(u)
        loss = sim_loss * self.sim_loss_weight + reg_loss * self.reg_loss_weight
        return {"sim_loss": sim_loss, "reg_loss": reg_loss, "loss": loss}


def l2reg_loss(u):
    """L2 regularisation loss"""
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def bending_energy_loss(u):
    """Bending energy regularisation loss"""
    derives = []
    ndim = u.size()[1]
    # 1st order
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    # 2nd order
    derives2 = []
    for i in range(ndim):
        derives2 += [finite_diff(derives[i], dim=i)]  # du2xx, du2yy, (du2zz)
    derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=1)]  # du2dxy
    if ndim == 3:
        derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=2)]  # du2dxz
        derives2 += [math.sqrt(2) * finite_diff(derives[1], dim=2)]  # du2dyz

    assert len(derives2) == 2 * ndim
    loss = torch.cat(derives2, dim=1).pow(2).sum(dim=1).mean()
    return loss


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Input shape (N, ndim, *sizes), mode='foward', 'backward' or 'central'"""
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
            raise ValueError(f"Mode {mode} not recognised")

        # reverse and join sublists into a flat list (Pytorch uses last -> first dim order)
        paddings.reverse()
        paddings = [p for ppair in paddings for p in ppair]

        # pad data
        if boundary == "Neumann":
            # Neumann boundary condition
            x_pad = F.pad(x, paddings, mode="replicate")
        elif boundary == "Dirichlet":
            # Dirichlet boundary condition
            x_pad = F.pad(x, paddings, mode="constant")
        else:
            raise ValueError("Boundary condition not recognised.")

        # slice and subtract
        x_diff = x_pad.index_select(
            dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)
        ) - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

        return x_diff
