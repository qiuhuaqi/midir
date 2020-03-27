"""Utility functions for data loading"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import functional as F

from model.submodules import spatial_transform
from utils.image import bbox_from_mask

"""OOP version to use with Torchvision Transforms"""
class CenterCrop(object):
    """
    Central crop numpy array
    Input shape: (N, H, W)
    Output shape: (N, H', W')
    """
    def __init__(self, output_size=192):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "'output_size' can only be a single integer or a pair of integers"
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[-2:]

        # pad to output size with zeros if image is smaller than crop size
        if h < self.output_size[0]:
            h_before = (self.output_size[0] - h) // 2
            h_after = self.output_size[0] - h - h_before
            image = np.pad(image, ((0 ,0), (h_before, h_after), (0, 0)), mode='constant')

        if w < self.output_size[1]:
            w_before = (self.output_size[1] - w) // 2
            w_after = self.output_size[1] - w - w_before
            image = np.pad(image, ((0, 0), (0, 0), (w_before, w_after)), mode='constant')

        # then continue with normal cropping
        h, w = image.shape[-2:]  # update shape numbers after padding
        h_start = h//2 - self.output_size[0]//2
        w_start = w//2 - self.output_size[1]//2

        h_end = h_start + self.output_size[0]
        w_end = w_start + self.output_size[1]

        cropped_image = image[..., h_start:h_end, w_start:w_end]

        assert cropped_image.shape[-2:] == self.output_size
        return cropped_image


class Normalise(object):
    """
    Normalise image of any shape to range
    (image - mean) / std
    mode:
        'minmax': normalise the image using its min and max to range [0, 1]
        'fixed': normalise the image by a fixed ration determined by the input arguments (preferred for image registration)
        'meanstd': normalise to mean=0, std=1
    """

    def __init__(self, mode='minmax',
                 min_in=0.0, max_in=255.0,
                 min_out=0.0, max_out=1.0):
        self.mode = mode
        self.min_in = min_in,
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

        if self.mode == 'fixed':
            self.norm_ratio = (max_out - min_out) * (max_in - min_in)

    def __call__(self, image, thres=(.05, 99.95)):

        # intensity clipping
        clip_min, clip_max = np.percentile(image, thres)
        image_clipped = image.copy()
        image_clipped[image < clip_min] = clip_min
        image_clipped[image > clip_max] = clip_max
        image = image_clipped  # re-assign reference

        if self.mode == 'minmax':  # determine the input min-max from input
            min_in = image.min()
            max_in = image.max()
            image_norm = (image - min_in) * (self.max_out - self.min_out) / (max_in - min_in + 1e-5)

        elif self.mode == 'fixed':  # use a fixed ratio
            image_norm = image * self.norm_ratio

        elif self.mode == 'meanstd':
            image_norm = (image - image.mean()) / image.std()

        else:
            raise ValueError("Normalisation mode not recogonised.")
        return image_norm


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image)



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
    dvf = gaussian_filter_fn(dvf_sparse)
    assert dvf.shape == dvf_sparse.shape, "DVF shape changed after Guassian smoothing"
    """"""

    """Mask the DVF with ROI bounding box"""
    bbox, mask_bbox_mask = bbox_from_mask(roi_mask, pad_ratio=bbox_pad_ratio)
    dvf *= np.expand_dims(mask_bbox_mask, 1)  # mask is expanded to (Nx1xHxW)

    # (future work) define the active region
    # active_region_i = (max(0, bbox_i[0] - 4*sigma + 1), min(image_shape[1], bbox_i[1] + 4*sigma - 1))
    # active_region_k = (max(0, bbox_k[0] - 4*sigma + 1), min(image_shape[3], bbox_k[1] + 4*sigma - 1))
    """"""
    # deform image
    dvf = dvf * 2 / dvf.shape[-1] # normalise DVF to pytorch coordinate space
    image_deformed = spatial_transform(torch.from_numpy(image).unsqueeze(1),  # (Nx1xHxW)
                                       torch.from_numpy(dvf))

    image_deformed = image_deformed.squeeze(1).numpy()  # (NxHxW)
    dvf = dvf * dvf.shape[-1] / 2  # reverse normalisation

    return image_deformed, dvf, mask_bbox_mask