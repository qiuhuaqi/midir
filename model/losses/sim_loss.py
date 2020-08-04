import math

import torch
from torch import nn as nn

from model.losses import window_func
from utils.image import normalise_intensity


class MILossBSpline(nn.Module):
    def __init__(self,
                 target_min=0.0,
                 target_max=1.0,
                 source_min=0.0,
                 source_max=1.0,
                 num_bins_target=32,
                 num_bins_source=32,
                 c_target=1.0,
                 c_source=1.0,
                 normalised=True,
                 window="cubic_bspline",
                 debug=False):

        super().__init__()
        self.debug = debug
        self.normalised = normalised
        self.window = window

        self.target_min = target_min
        self.target_max = target_max
        self.source_min = source_min
        self.source_max = source_max

        # set bin edges (assuming image intensity is normalised to the range)
        self.bin_width_target = (target_max - target_min) / num_bins_target
        self.bin_width_source = (source_max - source_min) / num_bins_source

        bins_target = torch.arange(num_bins_target) * self.bin_width_target + target_min
        bins_source = torch.arange(num_bins_source) * self.bin_width_source + source_min

        self.bins_target = bins_target.unsqueeze(1).float()  # (N, 1, #bins, H*W)
        self.bins_source = bins_source.unsqueeze(1).float()  # (N, 1, #bins, H*W)

        self.bins_target.requires_grad_(False)
        self.bins_source.requires_grad_(False)


        # determine kernel width controlling parameter (eps)
        # to satisfy the partition of unity constraint, eps can be no larger than bin_width
        # and can only be reduced by integer factor to increase kernel support,
        # cubic B-spline function has support of 4*eps, hence maximum number of bins a sample can affect is 4c
        self.eps_target = c_target * self.bin_width_target
        self.eps_source = c_source * self.bin_width_source

        if self.debug:
            self.histogram_joint = None
            self.p_joint = None
            self.p_target = None
            self.p_source = None

    @staticmethod
    def _parzen_window_1d(x, window="cubic_bspline"):
        if window == "cubic_bspline":
            return window_func.cubic_bspline_torch(x)
        elif window == "rectangle":
            return window_func.rect_window_torch(x)
        else:
            raise Exception("Window function not recognised.")

    def forward(self,
                target,
                source):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            target: (torch.Tensor, size (N, dim, *size)
            source: (torch.Tensor, size (N, dim, *size)

        Returns:
            (Normalise)MI: (scalar)
        """

        """pre-processing"""
        # normalise intensity for histogram calculation
        target = normalise_intensity(target[:, 0, ...],
                                     mode='minmax', min_out=self.target_min, max_out=self.target_max).unsqueeze(1)
        source = normalise_intensity(source[:, 0, ...],
                                     mode='minmax', min_out=self.source_min, max_out=self.source_max).unsqueeze(1)

        # flatten images to (N, 1, prod(*size))
        target = target.view(target.size()[0], target.size()[1], -1)
        source = source.view(source.size()[0], source.size()[1], -1)

        """histograms"""
        # bins to device
        self.bins_target = self.bins_target.type_as(target)
        self.bins_source = self.bins_source.type_as(source)

        # calculate Parzen window function response
        D_target = (self.bins_target - target) / self.eps_target
        D_source = (self.bins_source - source) / self.eps_source
        W_target = self._parzen_window_1d(D_target, window=self.window)  # (N, #bins, H*W)
        W_source = self._parzen_window_1d(D_source, window=self.window)  # (N, #bins, H*W)

        # calculate joint histogram (using batch matrix multiplication)
        hist_joint = W_target.bmm(W_source.transpose(1, 2))  # (N, #bins, #bins)
        hist_joint /= self.eps_target * self.eps_source

        """distributions"""
        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1)  # normalisation factor per-image,
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, target bins in dim1, source bins in dim2
        p_target = torch.sum(p_joint, dim=2)
        p_source = torch.sum(p_joint, dim=1)

        """entropy"""
        # calculate entropy
        ent_target = - torch.sum(p_target * torch.log(p_target + 1e-12), dim=1)  # (N,1)
        ent_source = - torch.sum(p_source * torch.log(p_source + 1e-12), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-12), dim=(1, 2))  # (N,1)

        """debug mode: store bins, histograms & distributions"""
        if self.debug:
            self.histogram_joint = hist_joint
            self.p_joint = p_joint
            self.p_target = p_target
            self.p_source = p_source

        # return (unnormalised) mutual information or normalised mutual information (NMI)
        if self.normalised:
            return -torch.mean((ent_target + ent_source) / ent_joint)
        else:
            return -torch.mean(ent_target + ent_source - ent_joint)


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    Adapting AirLab implementation to handle batches (& changing input):
    https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/loss/pairwise.py#L275
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=32,
                 sigma=0.5,
                 normalised=True,
                 debug=False):
        super(MILossGaussian, self).__init__()
        self.normalised = normalised

        self.vmin = vmin
        self.vmax = vmax

        # configure sigmas in number of bins
        self.sigma = sigma * (vmax - vmin) / num_bins

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

        self.debug = debug
        if self.debug:
            self.p_tar = None
            self.p_src = None
            self.p_joint = None

    def _compute_joint_prob(self, tar, src):
        """Compute joint distribution and entropy"""
        # flatten images to (N, 1, prod(*size))
        tar = tar.view(tar.size()[0], tar.size()[1], -1)
        src = src.view(src.size()[0], src.size()[1], -1)

        # cast bins
        self.bins = self.bins.type_as(tar)

        # calculate Parzen window function response (N, #bins, H*W)
        windowed_tar = torch.exp( -(tar - self.bins)**2 / (2 * self.sigma**2) ) / math.sqrt(2*math.pi) * self.sigma
        windowed_src = torch.exp( -(src - self.bins)**2 / (2 * self.sigma**2) ) / math.sqrt(2*math.pi) * self.sigma

        # calculate joint histogram batch
        hist_joint = windowed_tar.bmm(windowed_src.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1)  # normalisation per-image
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, tar, src):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            tar: (torch.Tensor, size (N, dim, *size)
            src: (torch.Tensor, size (N, dim, *size)

        Returns:
            (Normalise)MI: (scalar)
        """
        # normalise intensity for histogram calculation
        tar = normalise_intensity(tar[:, 0, ...],
                                  mode='minmax', min_out=self.vmin, max_out=self.vmax).unsqueeze(1)
        src = normalise_intensity(src[:, 0, ...],
                                  mode='minmax', min_out=self.vmin, max_out=self.vmax).unsqueeze(1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(tar, src)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, target bins in dim1, source bins in dim2
        p_tar = torch.sum(p_joint, dim=2)
        p_src = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_tar = - torch.sum(p_tar * torch.log(p_tar + 1e-10), dim=1)  # (N,1)
        ent_src = - torch.sum(p_src * torch.log(p_src + 1e-10), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-10), dim=(1, 2))  # (N,1)

        if self.debug:
            self.p_tar = p_tar
            self.p_src = p_src
            self.p_joint = p_joint

        if self.normalised:
            return -torch.mean((ent_tar + ent_src) / ent_joint)
        else:
            return -torch.mean(ent_tar + ent_src - ent_joint)
