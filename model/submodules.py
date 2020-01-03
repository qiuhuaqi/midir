"""Submodules to build the network"""
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


# ------------------------------------------- #
# BaseNet submodules
# ------------------------------------------- #
def relu():
    return nn.ReLU(inplace=True)


def conv_block_1(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity=relu):
    """Conv2d + Non-linearity + BN2d, Xavier initialisation"""
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    nn.init.xavier_uniform(conv_layer.weight, gain=np.sqrt(2.0))

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)
    # nn.init.constant_(bn_layer.weight, 1)
    # nn.init.constant_(bn_layer.bias, 0)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=1):
    """Block of 2x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride = strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=1):
    """Block of 3x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride = strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    conv3 = conv_block_1(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)



# ------------------------------------------- #
# Spatial Transformer Modules
# ------------------------------------------- #
def resample_transform(source, offset, interp='bilinear'):
    """
    Transform an image by sampling at coordinates on a deformed mesh grid.

    Args:
        source: source image, Tensor of shape (N, Ch, H, W)
        offset: deformation field from target to source, Tensor of shape (N, 2, H, W)
        interp: method of interpolation

    Returns:
        source image deformed using the deformation flow field,
        Tensor of the same shape as source image

    """

    # generate standard mesh grid
    h, w = source.size()[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])

    # stop autograd from calculating gradients on standard grid line
    grid_h = grid_h.requires_grad_(requires_grad=False).cuda()
    grid_w = grid_w.requires_grad_(requires_grad=False).cuda()

    # (N, 2, H, W) -> (N, 1, H, W) x 2
    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_h = offset_h.squeeze(1)
    offset_w = offset_w.squeeze(1)

    # (h,w) + (N, h, w) add by broadcasting
    grid_h = grid_h + offset_h
    grid_w = grid_w + offset_w

    # each pair of coordinates on deformed grid is using x-y order,
    # i.e. (column_num, row_num)
    # as required by the the grid_sample() function
    deformed_grid= torch.stack((grid_w, grid_h), 3)  # shape (N, H, W, 2)
    deformed_image = F.grid_sample(source, deformed_grid, mode=interp)

    return deformed_image


def resample_transform_cpu(source, offset, interp='bilinear'):
    """
    Transform an image by sampling at coordinates on a deformed mesh grid. CPU version.

    Args:
        source: source image, Tensor of shape (N, Ch, H, W)
        offset: deformation field from target to source, Tensor of shape (N, 2, H, W)
        interp: method of interpolation

    Returns:
        deformed_image: source image deformed using the deformation flow field,
                        Tensor of shape (N, Ch, H, W)

    """

    # generate standard mesh grid
    h, w = source.size()[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])

    # stop autograd from calculating gradients on standard grid line
    grid_h = grid_h.requires_grad_(requires_grad=False)
    grid_w = grid_w.requires_grad_(requires_grad=False)

    # (N, 2, H, W) -> (N, 1, H, W) x 2
    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_h = offset_h.squeeze(1)
    offset_w = offset_w.squeeze(1)

    # (h,w) + (N, h, w) add by broadcasting
    grid_h = grid_h + offset_h
    grid_w = grid_w + offset_w

    # each pair of coordinates on deformed grid is using x-y order,
    # i.e. (column_num, row_num)
    # as required by the the grid_sample() function
    deformed_grid= torch.stack((grid_w, grid_h), 3)  # shape (N, H, W, 2)
    deformed_image = F.grid_sample(source, deformed_grid, mode=interp)

    return deformed_image

