"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.image import normalise_intensity


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
        params: (object) parameters from params.json

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """
    sim_losses = {"MSE": nn.MSELoss(),
                  "NMI": MILoss()}
    reg_losses = {"diffusion": diffusion_loss,
                  "be": bending_energy_loss}

    sim_loss = sim_losses[params.sim_loss](data_dict["target"], data_dict["warped_source"]) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](data_dict["dvf_pred"]) * params.reg_weight

    return {"loss": sim_loss + reg_loss,
            params.sim_loss: sim_loss,
            params.reg_loss: reg_loss}



""" Regularisation loss """
from utils.spatial_diff import finite_diff

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


""" Regularity loss based on Jacobian """
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




""" Similarity loss """
from model import window_func

class MILoss(nn.Module):
    def __init__(self,
                 target_min=0.0,
                 target_max=1.0,
                 source_min=0.0,
                 source_max=1.0,
                 num_bins_target=64,
                 num_bins_source=64,
                 c_target=1.0,
                 c_source=1.0,
                 nmi=True,
                 window="cubic_bspline",
                 debug=False):

        super().__init__()
        self.debug = debug
        self.nmi = nmi
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
            (N)MI: (scalar)
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
        self.bins_target = self.bins_target.to(device=target.device)
        self.bins_source = self.bins_source.to(device=source.device)

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
        if self.nmi:
            return -torch.mean((ent_target + ent_source) / ent_joint)
        else:
            return -torch.mean(ent_target + ent_source - ent_joint)

""""""

