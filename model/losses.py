"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import normalise_intensity

"""
Construct the loss function (similarity + regularisation)
"""
def loss_fn(data_dict, params):
    """
    Unsupervised loss function

    Args:
        data_dict: (dict) dictionary containing data
            {
                "target": (Tensor, shape (N, ch, *dims)) target image
                "warped_source": (Tensor, shape (N, ch, *dims)) deformed source image
                "dvf_pred": (Tensor, shape (N, dim, *dims)) DVF predicted
                ...
            }
        params: (object) parameters from params.json

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """
    # todo: allow extra parameters to be passed to loss functions
    #  (e.g. NMI number of bins) via MILoss(*args, **kwargs)
    sim_losses = {"MSE": nn.MSELoss(),
                  "NMI": MILoss()}
    reg_losses = {"huber_spt": huber_loss_spatial,
                  "huber_temp": huber_loss_temporal,
                  "diffusion": diffusion_loss,
                  "be": bending_energy_loss}

    sim_loss = sim_losses[params.sim_loss](data_dict["target"], data_dict["warped_source"]) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](data_dict["dvf_pred"]) * params.reg_weight

    return {"loss": sim_loss + reg_loss,
            params.sim_loss: sim_loss,
            params.reg_loss: reg_loss}




""" Regularisation loss """
def diffusion_loss(dvf):
    """
    Compute diffusion regularisation on DVF

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field

    Returns:
        diffusion_loss_2d: (Scalar) diffusion regularisation loss
    """

    # boundary handling with padding to ensure all points are regularised
    # consideration of forward differences dx[i] = x[i+1] - x[i]
    # Neumann boundary conditions
    dvf_padx = F.pad(dvf, (0, 0, 0, 1), mode='replicate')  # pad H by 1 after, (N, 2, H+1, W)
    dvf_pady = F.pad(dvf, (0, 1, 0, 0), mode='replicate')  # pad W by 1 after, (N, 2, H, W+1)

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
    # consideration of forward differences dx[i] = x[i+1] - x[i]
    # Neumann boundary conditions
    dvf_padx = F.pad(dvf, (0, 0, 0, 1), mode='replicate')  # pad H by 1 before, (N, 2, H+1, W)
    dvf_pady = F.pad(dvf, (0, 1, 0, 0), mode='replicate')  # pad W by 1 before, (N, 2, H, W+1)

    dvf_dx = dvf_padx[:, :, 1:, :] - dvf_padx[:, :, :-1, :]  # (N, 2, H, W)
    dvf_dy = dvf_pady[:, :, :, 1:] - dvf_pady[:, :, :, :-1]  # (N, 2, H, W)


    # 2nd order derivatives
    # consideration of backward differences dxx[i] = dx[i] - dx[i-1]
    # Dirichlet boundary conditions
    dvf_dx_padx = F.pad(dvf_dx, (0, 0, 1, 0), mode='replicate')  # (N, 2, H+1, W)
    dvf_dx_pady = F.pad(dvf_dx, (1, 0, 0, 0), mode='replicate')  # (N, 2, H, W+1)
    dvf_dy_padx = F.pad(dvf_dy, (0, 0, 1, 0), mode='replicate')  # (N, 2, H+1, W)
    dvf_dy_pady = F.pad(dvf_dy, (1, 0, 0, 0), mode='replicate')  # (N, 2, H, W+1)

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
""""""




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

