import numpy as np
import torch
import torch.nn.functional as F
from utils.transformation import normalise_dvf
from utils.misc import param_ndim_setup

from model.mirtk_torch import cubic_bspline1d
from model.mirtk_torch import conv1d


# Transformation models #

class MultiResBSplineFFDTransform(object):
    def __init__(self, dim, img_size, lvls, cps):
        """
        Multi-resolution B-spline transformation

        Args:
            dim: (int) transformation model dimension
            img_size: (int or tuple)
            lvls: (int) number of multi-resolution levels
            cps: (int) control point spacing at the highest resolution
        """
        self.transforms = []
        for l in range(lvls):
            img_size_l = [imsz // (2 ** l) for imsz in img_size]
            cps_l = cps * (2 ** l)
            transform_l = CubicBSplineTransform(dim, img_size=img_size_l, cps=cps_l)
            self.transforms += [transform_l]

    def __call__(self, x):
        assert len(x) == len(self.transforms)
        dvfs = []
        for x_l, transform_l in zip(x, self.transforms):
            dvf_l = transform_l(x_l)
            dvfs += [dvf_l]
        return dvfs


class CubicBSplineTransform(object):
    def __init__(self, ndim, img_size=192, cps=5):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.

        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            order: (int) B-spline order
        """
        self.ndim = ndim
        self.img_size = param_ndim_setup(img_size, self.ndim)
        self.stride = param_ndim_setup(cps, self.ndim)

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2 for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def __call__(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            dvf: (N, dim, *(img_sizes)) The dense Displacement Vector Field of the transformation
        """
        # separable 1d transposed convolution
        y = x
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            y = conv1d(y, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, y.shape[0]), slice(0, y.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        y = y[slicer]
        return y


# class BSplineTransform(object):
#     def __init__(self, dim, img_size=192, cps=5, order=3):
#         """
#         Compute dense displacement field of B-spline FFD transformation model from input control point parameters.
#         B-spline kernel is computed by recursive convolutions.
#
#         Args:
#             dim: (int) image dimension
#             img_size: (int or tuple) size of the image, if int assume same size on all dimensions
#             cps: (int or tuple) control point spacing
#             order: (int) B-spline order
#         """
#         self.dim = dim
#         self.img_size = param_ndim_setup(img_size, self.dim)
#         self.cps = param_ndim_setup(cps, self.dim)
#         self.conv_transposed_fn = getattr(F, f"conv_transpose{self.dim}d")
#         self._set_kernel(order=order)
#
#         # TODO: configure stride and padding for the transposed convolution to perform the transformation computation correctly
#         self.stride = self.cps
#         # self.padding = [int((ks-1)/2) for ks in self.kernel.size()[2:]]
#         self.padding = [ks // 2 for ks in self.kernel.size()[2:]]
#
#     def _set_kernel(self, order=3):
#         """
#         Compute B-spline kernel of arbitrary order using recursive convolution
#         Adapted from AirLab implementation:
#         https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/utils/kernelFunction.py#L258
#
#         Control point spacing is set to be the size of 0th order B-spline function
#         """
#         kernel_ones = torch.ones(1, 1, *self.cps)
#         kernel = kernel_ones
#         padding = np.array(self.cps) - 1
#
#         convNd_fn = getattr(F, f"conv{self.dim}d")
#         for i in range(order):
#             kernel = convNd_fn(kernel, kernel_ones, padding=(padding).tolist()) / np.prod(self.cps)
#
#         self.kernel = kernel.repeat(self.dim, 1, *(1,) * self.dim)  # (dim, 1, *(kernel_sizes))
#         self.kernel = self.kernel
#
#     def __call__(self, x):
#         """
#         Args:
#             x: (N, dim, *(sizes)) Control point parameters
#         Returns:
#             dvf: (N, dim, *(img_sizes)) The dense Displacement Vector Field of the transformation
#         """
#         self.kernel = self.kernel.to(dtype=x.dtype, device=x.device)
#
#         # compute the DVF of the FFD transformation by transposed convolution 2D/3D
#         dvf = self.conv_transposed_fn(x,
#                                       weight=self.kernel,
#                                       stride=self.stride,
#                                       padding=self.padding,
#                                       groups=self.dim
#                                       )
#
#         #  crop the output to image size
#         for i in range(self.dim):
#             assert dvf.size()[i + 2] >= self.img_size[i], \
#                 f"FFD output DVF size ({dvf.size()[i + 2]}) is smaller than image size ({self.img_size[i]}) at dimension {i}"
#             # crop_start = dvf.size()[i + 2] // 2 - self.img_size[i] // 2
#             # dvf = dvf.narrow(i+2, crop_start, self.img_size[i])
#             # Note: half support should be already taken off from padding
#             crop_start = (self.kernel.size()[i + 2] // 2) - self.cps[i]
#             dvf = dvf.narrow(i + 2, crop_start, self.img_size[i])
#         return dvf


class DVFTransform(object):
    """ Displacement field model """

    def __init__(self):
        pass

    def __call__(self, x):
        return x


# Spatial transformation functions #

def spatial_transform(x, dvf, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Note that the dvf should not be spatially normalised.

    Args:
        x: (Tensor float, shape (N, ch, H, W) or (N, ch, H, W, D)) image to be spatially transformed
        dvf: (Tensor float, shape (N, 2, H, W) or (N, 3, H, W, D) dense displacement vector field (DVF) in i-j-k order
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    dim = x.ndim - 2
    size = x.size()[2:]

    # normalise DVF to [-1, 1]
    dvf = normalise_dvf(dvf)

    # generate standard mesh grid
    mesh_grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(dvf) for i in range(dim)])
    mesh_grid = [mesh_grid[i].requires_grad_(False) for i in range(dim)]

    # apply displacements to each direction (N, *size)
    deformed_meshgrid = [mesh_grid[i] + dvf[:, i, ...] for i in range(dim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    deformed_meshgrid = [deformed_meshgrid[dim - 1 - i] for i in range(dim)]
    deformed_meshgrid = torch.stack(deformed_meshgrid, -1)  # (N, *size, dim)

    return F.grid_sample(x, deformed_meshgrid, mode=interp_mode, align_corners=False)


def ml_spatial_transform(x_pyr, dvfs, interp_mode='bilinear'):
    """ Multi-resolution spatial transformation"""
    assert len(x_pyr) == len(dvfs)
    warped_x_pyr = []
    for (x, dvf) in zip(x_pyr, dvfs):
        warped_x = spatial_transform(x, dvf, interp_mode=interp_mode)
        warped_x_pyr.append(warped_x)
    return warped_x_pyr
