"""Submodules to build the network"""
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


# ------------------------------------------- #
# VoxelMorph and UNet submodules
# ------------------------------------------- #

def conv_bn_leaky_relu(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


def deconv_leaky_relu(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )



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
def resample_transform(source, dvf, interp='bilinear'):
    """
    Transform an image by sampling at coordinates on a deformed mesh grid.

    Args:
        source: source image, Tensor of shape (N, Ch, H, W)
        dvf: (Tensor, Nx2xHxW) displacement vector field from target to source, in number of pixels
        interp: method of interpolation

    Returns:
        source image deformed using the deformation flow field,
        Tensor of the same shape as source image

    """

    # generate standard mesh grid
    H, W = source.size()[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])

    # stop autograd from calculating gradients on standard grid line
    grid_h = grid_h.requires_grad_(requires_grad=False).cuda()
    grid_w = grid_w.requires_grad_(requires_grad=False).cuda()

    # deformed grid coordinates in pixel
    # (H,W) + (N, H, W) add by broadcasting
    new_grid_h = grid_h + dvf[:, 0, ...]
    new_grid_w = grid_w + dvf[:, 1, ...]

    # each pair of coordinates on deformed grid is using x-y order,
    # i.e. (column_num, row_num)
    # as required by the the grid_sample() function
    deformed_grid = torch.stack((new_grid_w, new_grid_h), 3)  # shape (N, H, W, 2)
    deformed_image = F.grid_sample(source, deformed_grid, mode=interp)

    return deformed_image
