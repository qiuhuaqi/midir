import torch.nn as nn
from torch.nn import functional as F


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
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    _ConvNd = getattr(nn, f"Conv{dim}d")

    # default initialisation is Kaiming uniform
    # see doc of _ConvNd(): https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
    return _ConvNd(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding)


# TODO: wrap conv + relu


def avg_pool(dim, kernel_size=2):
    """ Average pooling module of the specified dimension """
    _AvgPoolNd = getattr(nn, f"AvgPool{dim}d")
    return _AvgPoolNd(kernel_size)


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        if ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      recompute_scale_factor=False
                      )
    return y