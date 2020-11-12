import torch

import numpy as np
import os
import nibabel as nib
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


def normalise_dvf(dvf):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes dvf.yaml size is the same as the corresponding image.
    # TODO: allow passing image space size

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


def warp(x, dvf, interp_mode="bilinear"):
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
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(dvf) for i in range(dim)])
    grid = [grid[i].requires_grad_(False) for i in range(dim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + dvf[:, i, ...] for i in range(dim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[dim - 1 - i] for i in range(dim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)


def multi_res_warp(x_pyr, dvfs, interp_mode='bilinear'):
    """ Multi-resolution spatial transformation"""
    assert len(x_pyr) == len(dvfs)
    warped_x_pyr = []
    for (x, dvf) in zip(x_pyr, dvfs):
        warped_x = warp(x, dvf, interp_mode=interp_mode)
        warped_x_pyr.append(warped_x)
    return warped_x_pyr


def svf_exp(flow, scale=1, steps=5, sampling='bilinear'):
    r"""Group exponential maps of flow fields computed using scaling and squaring.

    Args:
        flow: Batch of flow fields as tensor of shape ``(N, D, ..., X)``.
        scale: Constant flow field scaling factor.
        steps: Number of scaling and squaring steps.
        sampling: interpolation mode in warping
        padding: Flow field extrapolation mode.
        align_corners: Whether ``flow`` vectors are defined with respect to
            ``Domain.CUBE`` (False) or ``Domain.CUBE_CORNERS`` (True).

    Returns:
        Exponential map of input flow field. If ``steps=0``, a reference to ``flow`` is returned.

    """
    if scale is None:
        scale = 1
    if steps is None:
        steps = 5
    if not isinstance(steps, int):
        raise TypeError("expv() 'steps' must be of type int")

    if steps < 0:
        raise ValueError("expv() 'steps' must be positive value")
    if steps == 0:
        return flow

    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, dvf=disp,
                           interp_mode=sampling)
    return disp