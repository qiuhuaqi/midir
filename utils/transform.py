import torch
import torch.nn.functional as F

import numpy as np
import os
import nibabel as nib
from model.transformations import spatial_transform


def dvf_line_integral(op_flow):
    """
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


def dof_to_dvf(target_img, dofin, dvfout, output_dir):
    """
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
        dvf: (ndarray of shape HxWx2) dense DVF, numpy array coordinate system
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
    dvf = np.array([dvf_y, dvf_x]).transpose(1, 2, 0)  # (H, W, 2), notice the x-y swap

    # save flow to nifti
    if dvfout is not None:
        ndvf = nib.Nifti1Image(dvf, nim.affine)
        nib.save(ndvf, '{0}/{1}.nii.gz'.format(output_dir, dvfout))

    # clean up: remove all the mesh files
    os.system('rm {0}/*mesh*'.format(output_dir))

    return dvf


def spatial_transform_numpy(source_img, dvf):
    """
    Warp numpy array image using Pytorch's resample based function on CPU

    Args:
        source_img: (ndarry shape: H,W,N) source image, assume square image
        dvf: (ndarray shape: H,W,N,1,2) dense displacement vector field, not normalised to [-1,1]

    Returns:
        warped_source_img: (ndarray shape: H,W,N) resample deformed source image
    """

    dvf_norm = 2 * dvf / source_img.shape[0]  # normalise to Pytorch coordinate system
    dvf_tensor = torch.from_numpy(dvf_norm[:, :, :, 0, :].transpose(2, 3, 0, 1)).float()  # tensor (N, 2, H, W)
    source_img_tensor = torch.from_numpy(source_img.transpose(2, 0, 1)).unsqueeze(1)  # tensor (N, 1, H, W)
    warped_source_img_tensor = spatial_transform(source_img_tensor, dvf_tensor)
    warped_source_img = warped_source_img_tensor.numpy().transpose(2, 3, 0, 1)[..., 0]  # (H, W, N)

    return warped_source_img


def normalise_dvf(dvf):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes dvf size is the same as the corresponding image.

    Args:
        dvf: (numpy.ndarray or torch.Tensor, size (N, dim, *sizes) Displacement Vector Field

    Returns:
        dvf (normalised)
    """

    dim = len(dvf.shape) - 2

    if type(dvf) is np.ndarray:
        factors = 2. / np.array(dvf.shape[2:])
        factors = factors.reshape(1, dim, *(1,) * dim)

    elif type(dvf) is torch.Tensor:
        factors = 2. / torch.tensor(tuple(dvf.size()[2:]), dtype=dvf.dtype, device=dvf.device)
        factors = factors.view(1, dim, *(1,) * dim)

    else:
        raise RuntimeError("DVF normalisation: input data type not recognised. "
                           "Expect: numpy.ndarray or torch.Tensor")
    return dvf * factors



def denormalise_dvf(dvf):
    """
    Invert of `normalise_dvf()`
    Output DVF is in number of pixels/voxels
    """
    dim = len(dvf.shape) - 2

    if type(dvf) is np.ndarray:
        factors = np.array(dvf.shape[2:]) / 2.
        factors = factors.reshape(1, dim, *(1,) * dim)  # (1, dim, *(1,)*dim)

    elif type(dvf) is torch.Tensor:
        factors = torch.tensor(tuple(dvf.size()[2:]), dtype=dvf.dtype, device=dvf.device) / 2.
        factors = factors.view(1, dim, *(1,) * dim)  # (1, dim, *(1,)*dim)

    else:
        raise RuntimeError("DVF normalisation: input data type not recognised. "
                           "Expect: numpy.ndarray or torch.Tensor")
    return dvf * factors
