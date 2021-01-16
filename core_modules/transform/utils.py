import torch

import numpy as np
import os
import nibabel as nib
from torch import Tensor
from torch.nn import functional as F


def dof_to_dvf(target_img, dofin, dvfout, output_dir):
    """
    (Legacy 2D only)
    Convert MIRTK format DOF file to dense Deformation Vector Field (DVF)
    by warping meshgrid using MIRTK transform-image

    Note: this function only handles 2D deformation a.t.m.

    Args:
        target_img: (str) full path of target image of the registration
        dofin: (str) full path to input DOF file
        dofdir: (str) full path of the dir of input DOF file
        dvfout: (str) output DVF file name under output_dir (without ".nii.gz")
        output_dir: (str) full path of output dir

    Returns:
        dvf.yaml: (ndarray of shape HxWx2) dense DVF, numpy array coordinate system
            this DVF is in number of pixels (not normalised)

    """

    # read a target image to get the affine transformation
    # this is to ensure the mirtk transform-image apply the DOF in the correct target space
    nim = nib.load(target_img)

    # generate the meshgrids
    mesh_x, mesh_y = np.meshgrid(range(0, nim.shape[0], 1), range(0, nim.shape[1], 1))
    mesh_x = mesh_x.astype('float')
    mesh_y = mesh_y.astype('float')

    # save both (x and y) into nifti files, using the target image header
    nim_meshx = nib.Nifti1Image(mesh_x, nim.affine)
    meshx_path = '{0}/{1}_meshx.nii.gz'.format(output_dir, dvfout)
    nib.save(nim_meshx, meshx_path)

    nim_meshy = nib.Nifti1Image(mesh_y, nim.affine)
    meshy_path = '{0}/{1}_meshy.nii.gz'.format(output_dir, dvfout)
    nib.save(nim_meshy, meshy_path)

    # use mirtk to transform it with DOF file as input
    warped_meshx_path = '{0}/{1}_meshx_warped.nii.gz'.format(output_dir, dvfout)
    warped_meshy_path = '{0}/{1}_meshy_warped.nii.gz'.format(output_dir, dvfout)
    os.system('mirtk transform-image {0} {1} -dofin {2} -target {3}'.format(meshx_path, warped_meshx_path, dofin, target_img))
    os.system('mirtk transform-image {0} {1} -dofin {2} -target {3}'.format(meshy_path, warped_meshy_path, dofin, target_img))

    # read in the generated mesh grid x and grid y
    warp_meshx = nib.load(warped_meshx_path).get_data()[:, :, 0]
    warp_meshy = nib.load(warped_meshy_path).get_data()[:, :, 0]

    # calculate the DVF by substracting the initial mesh grid from it
    dvf_x = warp_meshx - mesh_x
    dvf_y = warp_meshy - mesh_y
    dvf = np.array([dvf_y, dvf_x])  # (2, H, W), notice the x-y swap

    # save flow to nifti
    if dvfout is not None:
        ndvf = nib.Nifti1Image(dvf.transpose(1, 2, 0), nim.affine)
        nib.save(ndvf, '{0}/{1}.nii.gz'.format(output_dir, dvfout))

    # clean up: remove all the mesh files
    os.system('rm {0}/*mesh*'.format(output_dir))

    return dvf


def normalise_disp(dvf):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes dvf.yaml size is the same as the corresponding image.

    Args:
        dvf: (numpy.ndarray or torch.Tensor, shape (N, dim, *size)) Displacement Vector Field

    Returns:
        dvf.yaml: (normalised DVF)
    """

    dim = dvf.ndim - 2

    if type(dvf) is np.ndarray:
        norm_factors = 2. / np.array(dvf.shape[2:])
        norm_factors = norm_factors.reshape(1, dim, *(1,) * dim)

    elif type(dvf) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(dvf.size()[2:], dtype=dvf.dtype, device=dvf.device)
        norm_factors = norm_factors.view(1, dim, *(1,)*dim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return dvf * norm_factors


def denormalise_dvf(dvf):
    """
    Invert of `normalise_dvf()`
    Output DVF is in number of pixels/voxels
    """
    dim = dvf.ndim - 2

    if type(dvf) is np.ndarray:
        denorm_factors = np.array(dvf.shape[2:]) / 2.
        denorm_factors = denorm_factors.reshape(1, dim, *(1,) * dim)

    elif type(dvf) is torch.Tensor:
        denorm_factors = torch.tensor(tuple(dvf.size()[2:]), dtype=dvf.dtype, device=dvf.device) / torch.tensor(2.)
        denorm_factors = denorm_factors.view(1, dim, *(1,) * dim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return dvf * denorm_factors


def dvf_line_integral(op_flow):
    """
    (Legacy)
    Perform approximated line integral of frame-to-frame optical flow
    using Pytorch and GPU

    Args:
        op_flow: optical flow, Tensor, shape (N, 2, W, H)

    Returns:
        accum_flow: line integrated optical flow, same shape as input
    """
    # generate a standard grid
    h, w = op_flow.size()[-2:]
    std_grid_h, std_grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])
    std_grid = torch.stack([std_grid_h, std_grid_w])  # (2, H, W)
    std_grid = std_grid.cuda().float()

    # loop over frames
    accum_flow = []
    for fr in range(op_flow.size()[-1]):
        if fr == 0:
            new_grid = std_grid + op_flow[fr, ...]  # (2, H, W) + (2, H, W)
        else:
            # sample optical flow at current new_grid, then update new_grid by sampled offset
            # this line is unnecessarily complicated thanks to pytorch
            new_grid += F.grid_sample(op_flow[fr, ...].unsqueeze(0), new_grid.unsqueeze(0).permute(0, 2, 3, 1)).squeeze(0)  # (2, H, W)
        accum_flow += [new_grid - std_grid]

    accum_flow = torch.stack(accum_flow)  # (N, 2, H, W)
    return accum_flow


def svf_exp(flow, scale=1, steps=5, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline1d(stride, derivative: int = 0, dtype=None, device= None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=torch.float)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def conv1d(
        data: Tensor,
        kernel: Tensor,
        dim: int = -1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        transpose: bool = False
) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    result = data.type(kernel.dtype)  # (n, ndim, h, w, d)
    result = result.transpose(dim, -1)  # (n, ndim, ..., shape[dim])
    shape_ = result.size()
    # use native pytorch
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    # groups = numel(shape_[1:-1])  # (n, nidim * shape[not dim], shape[dim])
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # 3*w*d, 1, kernel_size
    result = result.reshape(shape_[0], groups, shape_[-1])  # n, 3*w*d, shape[dim]
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(
        result,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = result.transpose(-1, dim)
    return result


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Note: disp should NOT be spatially normalised to [-1, 1] space

    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise Disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)


def multi_res_warp(x_pyr, disps, interp_mode='bilinear'):
    """ Multi-resolution spatial transformation"""
    assert len(x_pyr) == len(disps)
    warped_x_pyr = []
    for (x, disp) in zip(x_pyr, disps):
        warped_x = warp(x, disp, interp_mode=interp_mode)
        warped_x_pyr.append(warped_x)
    return warped_x_pyr
