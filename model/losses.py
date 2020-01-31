"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np

from model.submodules import resample_transform
from model.unflow_losses import _second_order_deltas

##############################################################################################
# --- Temp test loss functions --- #
##############################################################################################

def smooth_mse_loss(flow, target, source, params):
    """
    MSE loss with smoothness regularisation.

    Args:
        flow: (Tensor, shape Nx2xHxW) predicted flow from target image to source image
        target: (Tensor, shape NxchxHxW) target image
        source: (Tensor, shape NxchxHxW) source image

    Returns:
        loss
    """

    # warp the source image towards target using grid resample
    # i.e. flow is from target to source
    warped_source = resample_transform(source, flow)

    mse = nn.MSELoss()
    mse_loss = mse(target, warped_source)
    if params.loss_fn == 'smooth_2nd':
        smooth_loss = params.smooth_weight * vanilla_second_order_loss(flow)

    loss = mse_loss + smooth_loss
    losses = {'mse': mse_loss, 'smooth': smooth_loss}

    return loss, losses

def vanilla_second_order_loss(flow):
    """Compute sum of 2nd order derivatives of flow as smoothness loss"""
    delta_h, delta_w, inner_masks = _second_order_deltas(flow)
    inner_masks = inner_masks.cuda()  # mask out borders
    loss_h = torch.sum((delta_h**2) * inner_masks) / np.prod(delta_h.size())
    loss_w = torch.sum((delta_w**2) * inner_masks) / np.prod(delta_w.size())
    return loss_h + loss_w


def vanilla_mse_loss(flow, img1, img2):
    img2_warped = resample_transform(img2, flow)
    mse = nn.MSELoss()
    loss = mse(img1, img2_warped)
    return loss


##############################################################################################
# --- Regularisation loss --- #
##############################################################################################

def diffusion_loss(dvf):
    """
    Calculate diffusion loss as a regularisation on the displacement vector field (DVF)

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        diffusion_loss_2d: (Scalar) diffusion regularisation loss
        """

    # finite difference as derivative
    # (note the 1st column of dx and the first row of dy are not regularised)
    dvf_dx = dvf[:, :, 1:, 1:] - dvf[:, :, :-1, 1:]  # (N, 2, H-1, W-1)
    dvf_dy = dvf[:, :, 1:, 1:] - dvf[:, :, 1:, :-1]  # (N, 2, H-1, W-1)
    return (dvf_dx.pow(2) + dvf_dy.pow(2)).mean()


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
    eps = 0.0001  # numerical stability

    # magnitude of the flow
    dvf_norm = torch.norm(dvf, dim=1)  # (N, H, W)

    # temporal finite derivatives, 1st order
    dvf_norm_dt = dvf_norm[1:, :, :] - dvf_norm[:-1, :, :]
    loss = (dvf_norm_dt.pow(2) + eps).sum().sqrt()
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
    warped_source = resample_transform(source, dvf)

    sim_loss = sim_losses[params.sim_loss](target, warped_source)
    reg_loss = reg_losses[params.reg_loss](dvf) * params.reg_weight

    loss = sim_loss + reg_loss
    losses = {params.sim_loss: sim_loss,  params.reg_loss: reg_loss}

    return loss, losses

