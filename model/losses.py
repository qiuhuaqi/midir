"""
Loss functions
"""
import math

import torch
import torch.nn as nn

from utils.image import normalise_intensity, bbox_from_mask
from utils.spatial_diff import finite_diff
from model import window_func


def loss_fn(data_dict, params):
    """
    Construct loss function

    Args:
        data_dict: (dict) dictionary containing data
            {
                "target": (Tensor, shape (N, 1, *sizes)) target image
                "warped_source": (Tensor, shape (N, 1, *sizes)) deformed source image
                "dvf_pred": (Tensor, shape (N, dim, *sizes)) DVF predicted
                ...
            }
        params: (object) parameters from params_mirtk.json

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """
    sim_losses = {"MSE": nn.MSELoss(),
                  "NMI": MILossGaussian(num_bins_tar=params.mi_num_bins,
                                        num_bins_src=params.mi_num_bins,
                                        sigma_tar=params.mi_sigma,
                                        sigma_src=params.mi_sigma)}

    reg_losses = {"diffusion": diffusion_loss,
                  "be": bending_energy_loss}

    tar = data_dict["target"]
    warped_src = data_dict["warped_source"]

    # (optional) only evaluate similarity loss within ROI bounding box
    if params.loss_roi:
        bbox, _ = bbox_from_mask(data_dict["roi_mask"].squeeze(1).numpy())
        for i in range(params.dim):
            tar = tar.narrow(i+2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))
            warped_src = warped_src.narrow(i+2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))

    sim_loss = sim_losses[params.sim_loss](tar, warped_src) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](data_dict["dvf_pred"]) * params.reg_weight

    return {"loss": sim_loss + reg_loss,
            params.sim_loss: sim_loss,
            params.reg_loss: reg_loss}


def diffusion_loss(dvf):
    """
    Compute diffusion (L2) regularisation loss

    Args:
        dvf: (torch.Tensor, size (N, dim, *sizes)) Dense displacement vector field

    Returns:
        diffusion loss: (scalar) Diffusion (L2) regularisation loss
    """
    dvf_dxyz = finite_diff(dvf, mode="forward")
    return torch.cat(dvf_dxyz, dim=1).pow(2).sum(dim=1).mean()


def bending_energy_loss(dvf):
    """
    Compute the Bending Energy regularisation loss (Rueckert et al., 1999)

    Args:
        dvf: (torch.Tensor, size (N, dim, *sizes)) Dense displacement vector field

    Returns:
        BE: (scalar) Bending Energy loss
    """
    # 1st order derivatives
    dvf_d1 = finite_diff(dvf, mode="forward")

    # 2nd order derivatives
    dvf_d2 = []
    for dvf_d in dvf_d1:
        dvf_d2 += finite_diff(dvf_d, mode="forward")
    return torch.cat(dvf_d2, dim=1).pow(2).sum(dim=1).mean()


# def compute_jacobian(x):
#     """ reference code from Chen"""
#     bsize, csize, height, width = x.size()
#     # padding
#     v = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()),
#                   3)
#     u = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()),
#                   2)
#
#     d_x = (torch.index_select(v, 3, torch.arange(2, width + 2).cuda())
#            - torch.index_select(v, 3, torch.arange(width).cuda())) / 2
#     d_y = (torch.index_select(u, 2, torch.arange(2, height + 2).cuda()) - torch.index_select(u, 2, torch.arange(
#         height).cuda())) / 2
#
#     J = (torch.index_select(d_x, 1, torch.tensor([0]).cuda())+1)*(torch.index_select(d_y, 1, torch.tensor([1]).cuda())+1) \
#         -torch.index_select(d_x, 1, torch.tensor([1]).cuda())*torch.index_select(d_y, 1, torch.tensor([0]).cuda())
#     return J
#




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
                 tar_min=0.0, tar_max=1.0,
                 src_min=0.0, src_max=1.0,
                 num_bins_tar=32,
                 num_bins_src=32,
                 sigma_tar=0.5,
                 sigma_src=0.5,
                 normalised=True,
                 debug=False):
        super(MILossGaussian, self).__init__()
        self.normalised = normalised

        self.tar_min = tar_min
        self.tar_max = tar_max
        self.src_min = src_min
        self.src_max = src_max

        # input sigmas are in number of bins
        self.sigma_tar = sigma_tar * (tar_max - tar_min) / num_bins_tar
        self.sigma_src = sigma_src * (src_max - src_min) / num_bins_src

        # set bin edges
        self.bins_tar = torch.linspace(self.tar_min, self.tar_max, num_bins_tar, requires_grad=False).unsqueeze(1)
        self.bins_src = torch.linspace(self.src_min, self.src_max, num_bins_src, requires_grad=False).unsqueeze(1)

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
        self.bins_tar = self.bins_tar.type_as(tar)
        self.bins_src = self.bins_src.type_as(src)

        # calculate Parzen window function response (N, #bins, H*W)
        windowed_tar = torch.exp( -(tar - self.bins_tar)**2 / (2 * self.sigma_tar**2) ) / math.sqrt(2*math.pi) * self.sigma_tar
        windowed_src = torch.exp( -(src - self.bins_src)**2 / (2 * self.sigma_src**2) ) / math.sqrt(2*math.pi) * self.sigma_src

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
                                  mode='minmax', min_out=self.tar_min, max_out=self.tar_max).unsqueeze(1)
        src = normalise_intensity(src[:, 0, ...],
                                  mode='minmax', min_out=self.src_min, max_out=self.src_max).unsqueeze(1)

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
