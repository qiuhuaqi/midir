import math
import numpy as np
import torch
from torch.nn import functional as F

from model.transformations import spatial_transform
from utils.image import bbox_from_mask
from utils.transform import normalise_dvf, denormalise_dvf


"""Deformation synthesis using control point + Gaussian filter model"""
class GaussianFilter(object):
    def __init__(self, sigma, kernel_size=None, dim=2):
        """
        Using class to avoid repeated computation of Gaussian kernel
        This module expands filter the all dimensions assuming isotropic data
        """
        if not kernel_size:  # if not specified, kernel_size = [-4simga, +4sigma]
            self.kernel_size = 8 * sigma + 1
        kernel_sizes = [self.kernel_size] * dim
        sigmas = [sigma] * dim

        # Compute Gaussian kernel as the product of
        # the Gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_sizes])  # i-j order
        for size, sigma, meshgrid in zip(kernel_sizes, sigmas, meshgrids):
            mean = (size - 1) / 2  # odd number kernel_size
            kernel *= 1 / (sigma * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((meshgrid - mean) / sigma) ** 2 / 2)

        # normalise to sum of 1
        self.kernel_norm_factor = kernel.sum()
        self.kernel = kernel / self.kernel_norm_factor   # size (kernel_size) * dim

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

    def __call__(self, x):
        """
        Apply Gaussian filter using Pytorch convolution.
        Each channel is filtered independently

        Args:
            x: (ndarray, NxChxHxW)
        Returns:
            output: (ndarray, NxChxHxW) Gaussian filter smoothed input
        """
        # input to tensor
        x = torch.from_numpy(x)

        # Repeat kernel for all input channels
        num_channel = x.size()[1]
        self.kernel = self.kernel.view(1, 1, *self.kernel.size())
        self.kernel = self.kernel.repeat(num_channel, *[1] * (self.kernel.dim() - 1))  # (Ch, 1, *[1]*input.dim())
        self.kernel = self.kernel.type_as(x)

        with torch.no_grad():
            output = self.conv(x, weight=self.kernel, groups=num_channel, padding=int(self.kernel_size // 2))
        return output.numpy()


def synthesis_elastic_deformation(image,
                                  roi_mask,
                                  sigma=5,
                                  cps=10,
                                  disp_range=(0, 1),
                                  bbox_pad_ratio=0.2):
    """
    Generate synthetic random elastic deformation and images (2D)
    Using control point model & Gaussian kernel smoothing
    DVF is the same for all slices across N

    Args:
        image: (ndarray, NxHxW) input image
        roi_mask: (ndarray, NxHxW) Region of Interest (ROI) mask where positive value is > 0
        sigma: Gaussian filter std
        cps: control point spacing
        disp_range: (tuple) range of of number of maximum pixels/voxels displacement allowed at the control point
        bbox_pad_ratio: proportion of distance from the brain mask bbox to image boundary to pad

    Returns:
        image_deformed: (ndarray, NxHxW) elastically deformed input image
        dvf: (ndarray, Nx2xHxW) dense vector field of the deformation, in number of pixels not [-1,1]

    """
    """Generate random elastic DVF"""
    # randomly generate control point parameters for one 2D image
    cp_meshgrid = np.meshgrid(np.arange(0, image.shape[1], cps), np.arange(0, image.shape[2], cps), indexing='ij')
    num_cp_i, num_cp_j = cp_meshgrid[0].shape[0], cp_meshgrid[0].shape[1]  # number of control points

    # automatically find the control point scale range according to the max displacement range
    disp_range = np.array(disp_range) * 2*math.pi*(sigma**2)  # multiply by the constant factor
    cp_scale_i = np.random.uniform(disp_range[0], disp_range[1])
    cp_scale_j = np.random.uniform(disp_range[0], disp_range[1])

    cp_params_i = np.random.uniform(-1, 1, (num_cp_i, num_cp_j)) * cp_scale_i
    cp_params_j = np.random.uniform(-1, 1, (num_cp_i, num_cp_j)) * cp_scale_j

    # initialise dvf for one 2D image
    di = np.zeros(image.shape[1:], dtype=np.float32)
    dj = np.zeros(image.shape[1:], dtype=np.float32)  # (HxW)

    # fill in the control point values
    di[tuple(cp_meshgrid)] = cp_params_i
    dj[tuple(cp_meshgrid)] = cp_params_j

    # stack dvfs and repeat for all slices (todo: this behavior can be changed later)
    dvf_sparse_2D = np.array([di, dj])  # 2xHxW
    dvf_sparse = np.tile(dvf_sparse_2D, (image.shape[0], 1, 1, 1))  # Nx2xHxW

    # apply Gaussian smoothing
    gaussian_filter_fn = GaussianFilter(sigma=sigma)
    dvf = gaussian_filter_fn(dvf_sparse)  # (N, 2, H, W)
    assert dvf.shape == dvf_sparse.shape, "DVF shape changed after Guassian smoothing"
    """"""

    """Mask the DVF with ROI bounding box"""
    mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask, pad_ratio=bbox_pad_ratio)
    dvf *= mask_bbox_mask[:, np.newaxis, ...]  # (N, dim, (*dims)) * (N, 1, (*dims))

    # (future work) define the active region of the control point grid
    # active_region_i = (max(0, bbox_i[0] - 4*sigma + 1), min(image_shape[1], bbox_i[1] + 4*sigma - 1))
    # active_region_k = (max(0, bbox_k[0] - 4*sigma + 1), min(image_shape[3], bbox_k[1] + 4*sigma - 1))
    """"""

    # deform image
    dvf = normalise_dvf(dvf)
    image_deformed = spatial_transform(torch.from_numpy(image).unsqueeze(1),  # (Nx1xHxW)
                                       torch.from_numpy(dvf)).squeeze(1).numpy()  # (NxHxW)
    dvf = denormalise_dvf(dvf)

    return image_deformed, dvf