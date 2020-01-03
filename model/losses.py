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
# --- Huber loss --- #
##############################################################################################

def huber_loss_spatial(flow):
    """
    Calculate approximated spatial Huber loss
    Args:
        flow: optical flow estimated, Tensor of shape (N, 2, H, W)

    Returns:
        spatial_huber_loss: the loss value

    """
    eps = 0.01  # numerical stability

    # magnitude of the flow
    flow_norm = torch.norm(flow, dim=1)  # (N, H, W)

    # spatial finite derivative, 1st order
    flow_norm_dx = flow_norm[:, 1:, :] - flow_norm[:, :-1, :]  # (N, H-1, W)
    flow_norm_dy = flow_norm[:, :, 1:] - flow_norm[:, :, :-1]  # (N, H, W-1)

    # calculate the huber loss step by step
    # drop 1 on one dimension to match shapes to (N, H-1, W-1)
    spatial_huber_loss = flow_norm_dx[:, :, :-1] ** 2 + flow_norm_dy[:, :-1, :] ** 2  # ^2 + ^2: (N, H-1, W-1)
    spatial_huber_loss = torch.sqrt( eps + torch.sum( torch.sum(spatial_huber_loss, dim=1), dim=1) )  # sqrt(sum over space)
    spatial_huber_loss = torch.mean(spatial_huber_loss)  # mean over time (batch)

    return spatial_huber_loss


def huber_loss_temporal(flow):
    """
    Calculate approximated temporal Huber loss

    Args:
        flow: optical flow estimated, Tensor of shape (N, 2, H, W)

    Returns:
        temporal_huber_loss: the loss value

    """
    eps = 0.01  # numerical stability

    # magnitude of the flow
    flow_norm = torch.norm(flow, dim=1)  # (N, H, W)

    # temporal finite derivatives, 1st order
    flow_norm_dt = flow_norm[1:, :, :] - flow_norm[:-1, :, :]
    temporal_huber_loss = torch.sqrt(eps + torch.sum(flow_norm_dt ** 2))
    return temporal_huber_loss


def huber_loss_fn(flow, target, source, params):
    """
    Unsupervised loss function with optional pseudo-Huber loss as regularisation

    Args:
        flow: (Tensor, shape Nx2xHxW) predicted flow from target image to source image
        target: (Tensor, shape NxchxHxW) target image
        source: (Tensor, shape NxchxHxW) source image

    Returns:
        loss
        losses: (dict) dictionary of individual loss term after weighting
    """

    # warp the source image towards target using grid resample
    # i.e. flow is from target to source
    warped_source = resample_transform(source, flow)

    mse = nn.MSELoss()
    mse_loss = mse(target, warped_source)
    smooth_loss = params.huber_spatial * huber_loss_spatial(flow) + params.huber_temporal * huber_loss_temporal(flow)

    loss = mse_loss + smooth_loss
    losses = {'mse': mse_loss,  'smooth_loss': smooth_loss}

    return loss, losses


##############################################################################################
# --- mutual information loss --- #
##############################################################################################
from model.mutual_info.histogram import JointHistParzenTorch
from model.mutual_info.mutual_info import NMI_pytorch

def nmi_loss_fn(flow, target, source, params):
        """
        Unsupervised loss function based on normalised mutual information

        Args:
            flow: (Tensor, shape Nx2xHxW) predicted flow from target image to source image
            target: (Tensor, shape NxChxHxW) target image
            source: (Tensor, shape NxChxHxW) source image
            params: parameters in params.json

        Returns:
            loss
            losses: (dict) dictionary of individual loss term after weighting
        """

        # warp the source image towards target using grid resample
        # i.e. flow is from target to source
        warped_source = resample_transform(source, flow)

        # compute normalised mutual information
        joint_hist_fn = JointHistParzenTorch().cuda()
        # todo: find a better way to put to cuda (needed because of the bin edge array need to put on GPU) or do I need to if I'm using nn.functional?
        joint_hist = joint_hist_fn(target, warped_source)
        nmi = NMI_pytorch(joint_hist)

        # add regularisation (approximated Huber)
        smooth_loss = params.huber_spatial * huber_loss_spatial(flow) + params.huber_temporal * huber_loss_temporal(flow)
        # todo: bending energy

        # negative NMI loss function to minimise
        loss = - nmi + smooth_loss
        losses = {'nmi': nmi, 'smooth_loss': smooth_loss}

        return loss, losses

