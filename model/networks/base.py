import numpy as np
import torch.nn as nn

def conv_Nd(dim,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        dim: (int) dimension of the data/model

    Returns:
        (nn.Module instance)
    """
    _ConvNd = getattr(nn, f"Conv{dim}d")

    # default initialisation is Kaiming uniform
    # see class _ConvNd(): https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
    return _ConvNd(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding)


def avg_pool(dim, kernel_size=2):
    _AvgPoolNd = getattr(nn, f"AvgPool{dim}d")
    return _AvgPoolNd(kernel_size)


def conv_block_1(in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
    """Conv2d + Non-linearity + BN2d, Xavier initialisation"""
    conv_layer = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False)
    nn.init.xavier_uniform_(conv_layer.weight, gain=np.sqrt(2.0))

    bn_layer = nn.BatchNorm2d(out_channels)
    nll_layer = nn.ReLU(inplace=True)
    return nn.Sequential(conv_layer, bn_layer, nll_layer)


def conv_blocks_2(in_channels,
                  out_channels,
                  strides=1):
    """Block of 2x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    return nn.Sequential(conv1, conv2)


def conv_blocks_3(in_channels,
                  out_channels,
                  strides=1):
    """Block of 3x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    conv3 = conv_block_1(out_channels, out_channels, stride=1)
    return nn.Sequential(conv1, conv2, conv3)
