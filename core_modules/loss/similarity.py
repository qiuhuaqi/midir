import math
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from utils.misc import param_ndim_setup


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    Adapting AirLab implementation to handle batches (& changing input):
    https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/loss/pairwise.py#L275
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 threshold_roi=False,
                 threshold=0.0001,
                 normalised=True
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

        # self.threshold_roi = threshold_roi
        # self.threshold = threshold
        self.sample_ratio = sample_ratio

        self.normalised = normalised

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
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-10
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
        # batch size in dim0, target bins in dim1, source bins in dim2
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
    Modified on VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, tar, src):
        # products and squares
        tar2 = tar * tar
        src2 = src * src
        tar_src = tar * src

        # set window size
        ndim = tar.dim() - 2
        window_size = param_ndim_setup(self.window_size, ndim)

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(tar)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        tar_sum = conv_fn(tar, sum_filt, stride=stride, padding=padding)
        src_sum = conv_fn(src, sum_filt, stride=stride, padding=padding)
        tar2_sum = conv_fn(tar2, sum_filt, stride=stride, padding=padding)
        src2_sum = conv_fn(src2, sum_filt, stride=stride, padding=padding)
        tar_src_sum = conv_fn(tar_src, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        mu_tar = tar_sum / window_num_points
        mu_src = src_sum / window_num_points

        cov = tar_src_sum - mu_src * tar_sum - mu_tar * src_sum + mu_tar * mu_src * window_num_points
        tar_var = tar2_sum - 2 * mu_tar * tar_sum + mu_tar * mu_tar * window_num_points
        src_var = src2_sum - 2 * mu_src * src_sum + mu_src * mu_src * window_num_points

        lncc = cov * cov / (tar_var * src_var + 1e-5)

        return -torch.mean(lncc)

