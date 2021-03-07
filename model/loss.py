import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.misc import param_ndim_setup


class LossFn(nn.Module):
    def __init__(self,
                 sim_loss_fn,
                 reg_loss_fn,
                 sim_loss_weight=1.,
                 reg_loss_weight=1.
                 ):
        super(LossFn, self).__init__()
        self.sim_loss_fn = sim_loss_fn
        self.sim_loss_weight=sim_loss_weight
        self.reg_loss_fn = reg_loss_fn
        self.reg_loss_weight = reg_loss_weight

    def forward(self, tar, warped_src, u):
        sim_loss = self.sim_loss_fn(tar, warped_src)
        reg_loss = self.reg_loss_fn(u)
        loss = sim_loss * self.sim_loss_weight + reg_loss * self.reg_loss_weight
        return {'sim_loss': sim_loss,
                'reg_loss': reg_loss,
                'loss': loss}


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)


class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        window_size = param_ndim_setup(self.window_size, ndim)

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)


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
