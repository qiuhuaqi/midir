"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import normalise_torch

##############################################################################################
# --- Regularisation loss --- #
##############################################################################################

def diffusion_loss(dvf):
    """
    Compute diffusion regularisation on DVF

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field

    Returns:
        diffusion_loss_2d: (Scalar) diffusion regularisation loss
    """

    # boundary handling with padding to ensure all points are regularised
    dvf_padx = F.pad(dvf, (0, 0, 1, 0))  # pad H by 1 before, (N, 2, H+1, W)
    dvf_pady = F.pad(dvf, (1, 0, 0, 0))  # pad W by 1 before, (N, 2, H, W+1)

    dvf_dx = dvf_padx[:, :, 1:, :] - dvf_padx[:, :, :-1, :]  # (N, 2, H, W)
    dvf_dy = dvf_pady[:, :, :, 1:] - dvf_pady[:, :, :, :-1]  # (N, 2, H, W)
    return (dvf_dx.pow(2) + dvf_dy.pow(2)).mean()


def bending_energy_loss(dvf):
    """
    Bending Energy regularisation (Rueckert et al., 1999)

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field

    Returns:
        BE: (Scalar)
    """

    # 1st order derivatives
    # boundary handling with padding to ensure all points are regularised
    dvf_padx = F.pad(dvf, (0, 0, 1, 0))  # pad H by 1 before, (N, 2, H+1, W)
    dvf_pady = F.pad(dvf, (1, 0, 0, 0))  # pad W by 1 before, (N, 2, H, W+1)

    dvf_dx = dvf_padx[:, :, 1:, :] - dvf_padx[:, :, :-1, :]  # (N, 2, H, W)
    dvf_dy = dvf_pady[:, :, :, 1:] - dvf_pady[:, :, :, :-1]  # (N, 2, H, W)


    # 2nd order derivatives
    dvf_dx_padx = F.pad(dvf_dx, (0, 0, 1, 0))  # (N, 2, H+1, W)
    dvf_dx_pady = F.pad(dvf_dx, (1, 0, 0, 0))  # (N, 2, H, W+1)
    dvf_dy_padx = F.pad(dvf_dy, (0, 0, 1, 0))  # (N, 2, H+1, W)
    dvf_dy_pady = F.pad(dvf_dy, (1, 0, 0, 0))  # (N, 2, H, W+1)

    dvf_dxdx = dvf_dx_padx[:, :, 1:, :] - dvf_dx_padx[:, :, :-1, :]  # (N, 2, H, W)
    dvf_dxdy = dvf_dx_pady[:, :, :, 1:] - dvf_dx_pady[:, :, :, :-1]  # (N, 2, H, W)
    dvf_dydx = dvf_dy_padx[:, :, 1:, :] - dvf_dy_padx[:, :, :-1, :]  # (N, 2, H, W)
    dvf_dydy = dvf_dy_pady[:, :, :, 1:] - dvf_dy_pady[:, :, :, :-1]  # (N, 2, H, W)

    # print(dvf_dxdx.size(), dvf_dxdy.size(), dvf_dydx.size(), dvf_dydy.size())
    return (dvf_dxdx.pow(2).sum(dim=1) + dvf_dxdy.pow(2).sum(dim=1) + dvf_dydx.pow(2).sum(dim=1) + dvf_dydy.pow(2).sum(dim=1)).mean()


def huber_loss_spatial(dvf):
    """
    Calculate approximated spatial Huber loss
    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        loss: (Scalar) Huber loss spatial

    """
    eps = 0.0001 # numerical stability

    # finite difference as derivative
    # (note the 1st column of dx and the first row of dy are not regularised)
    dvf_dx = dvf[:, :, 1:, 1:] - dvf[:, :, :-1, 1:]  # (N, 2, H-1, W-1)
    dvf_dy = dvf[:, :, 1:, 1:] - dvf[:, :, 1:, :-1]  # (N, 2, H-1, W-1)
    return ((dvf_dx.pow(2) + dvf_dy.pow(2)).sum(dim=1) + eps).sqrt().mean()


def huber_loss_temporal(dvf):
    """
    Calculate approximated temporal Huber loss

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        loss: (Scalar) huber loss temporal

    """
    # todo: temporally regularise dx and dy not only the L2norm
    # magnitude of the flow
    dvf_norm = torch.norm(dvf, dim=1)  # (N, H, W)

    # temporal finite derivatives, 1st order
    dvf_norm_dt = dvf_norm[1:, :, :] - dvf_norm[:-1, :, :]
    loss = (dvf_norm_dt.pow(2) + 1e-4).sum().sqrt()
    return loss


##############################################################################################
# --- Similarity loss --- #
##############################################################################################

from model import window_func


class NMILoss(nn.Module):
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

        bins_target = torch.arange(num_bins_target, requires_grad=False) * self.bin_width_target + target_min
        bins_source = torch.arange(num_bins_source, requires_grad=False) * self.bin_width_source + source_min
        self.bins_target = bins_target.float().unsqueeze(1)  # (N, 1, #bins, H*W)
        self.bins_source = bins_source.float().unsqueeze(1)  # (N, 1, #bins, H*W)

        # window width parameter (eps)
        # to satisfy the partition of unity constraint, eps can be no larger than bin_width
        # and can only be reduced by integer factor to increase kernel support,
        # cubic B-spline function has support of 4*eps, hence maximum number of bins a sample can affect is 4c
        # todo: what are c(s)? (Look at original Unser paper)
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

        """pre-processing"""
        # normalise intensity range of both images to [0, 1] and cast to float 32
        target = normalise_torch(target.float(), self.target_min, self.target_max)
        source = normalise_torch(source.float(), self.source_min, self.source_max)

        # flatten images to (N, 1, H*W)
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
        histogram_joint = W_target.bmm(W_source.transpose(1, 2))  # (N, #bins, #bins)
        histogram_joint /= self.eps_target * self.eps_source

        """distributions"""
        # normalise joint histogram to acquire joint distribution
        p_joint = histogram_joint / histogram_joint.sum()

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, target bins in dim1, source bins in dim2
        p_target = torch.sum(p_joint, dim=2)
        p_source = torch.sum(p_joint, dim=1)

        """entropy"""
        # calculate entropy
        entropy_target = - torch.sum(p_target * torch.log(p_target + 1e-12), dim=1)  # (N,1)
        entropy_source = - torch.sum(p_source * torch.log(p_source + 1e-12), dim=1)  # (N,1)
        entropy_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-12), dim=(1, 2))  # (N,1)

        """debug mode: store bins, histograms & distributions"""
        if self.debug:
            self.histogram_joint = histogram_joint
            self.p_joint = p_joint
            self.p_target = p_target
            self.p_source = p_source

        # return (unnormalised) mutual information or normalised mutual information (NMI)
        if self.nmi:
            return -torch.mean((entropy_target + entropy_source) / entropy_joint)
        else:
            return -torch.mean(entropy_target + entropy_source - entropy_joint)



"""
Construct the loss function (similarity + regularisation)
"""
sim_losses = {"MSE": nn.MSELoss(),
              "NMI": NMILoss()}
reg_losses = {"huber_spt": huber_loss_spatial,
              "huber_temp": huber_loss_temporal,
              "diffusion": diffusion_loss,
              "be": bending_energy_loss}


def loss_fn(dvf, target, warped_source, params):
    """
    Unsupervised loss function

    Args:
        target: (Tensor, shape NxchxHxW) target image
        warped_source: (Tensor, shape NxchxHxW) registered source image
        params: (object) model parameters

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """

    # todo: allow extra parameters to be passed to loss functions
    #  (e.g. NMI number of bins)

    sim_loss = sim_losses[params.sim_loss](target, warped_source) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](dvf) * params.reg_weight

    loss = sim_loss + reg_loss
    losses = {params.sim_loss: sim_loss,  params.reg_loss: reg_loss}
    return loss, losses
