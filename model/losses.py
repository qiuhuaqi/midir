"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.submodules import spatial_transform

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

## outdated huber losses
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
# --- mutual information loss --- #
##############################################################################################
from model.mutual_info.histogram import JointHistParzenTorch
from model.mutual_info.mutual_info import nmi_from_joint_entropy_pytorch

def nmi_loss(x, y):
    joint_hist_fn = JointHistParzenTorch().to(device=x.device)
    joint_hist = joint_hist_fn(x, y)
    nmi_loss = - nmi_from_joint_entropy_pytorch(joint_hist)
    return nmi_loss

##############################################################################################
# --- construct the loss function --- #
##############################################################################################
sim_losses = {"MSE": nn.MSELoss(),
              "NMI": nmi_loss}
reg_losses = {"huber_spt": huber_loss_spatial,
              "huber_temp": huber_loss_temporal,
              "diffusion": diffusion_loss}


def loss_fn(dvf, target, source, params):
    """
    Unsupervised loss function

    Args:
        dvf: (Tensor, shape Nx2xHxW) predicted displacement vector field
        target: (Tensor, shape NxchxHxW) target image
        source: (Tensor, shape NxchxHxW) source image
        params: (object) model parameters

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """

    # warp the source image towards target using grid resample (spatial transformer)
    # i.e. dvf is from target to source
    warped_source = spatial_transform(source, dvf)

    sim_loss = sim_losses[params.sim_loss](target, warped_source) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](dvf) * params.reg_weight

    loss = sim_loss + reg_loss
    losses = {params.sim_loss: sim_loss,  params.reg_loss: reg_loss}

    return loss, losses

