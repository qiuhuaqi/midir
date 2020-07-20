import math
import numpy as np
import torch
from torch.nn import functional as F

from model.transformations import spatial_transform
from utils.image import bbox_from_mask
from utils.misc import param_dim_setup


class GaussianFilter(object):
    """
    Gaussian Filter (nD)
    """
    def __init__(self,
                 dim,
                 sigma,
                 kernel_size=None,
                 device=torch.device('cpu')
                 ):
        self.dim = dim
        self.device = device

        # configure Gaussian kernel standard deviation (sigma) and kernel size
        sigmas = param_dim_setup(sigma, dim)
        if not kernel_size:
            # if not specified, kernel defined in [-4simga, +4sigma]
            kernel_size = [8 * sigmas[i] + 1
                           for i in range(dim)]
        else:
            kernel_size = param_dim_setup(kernel_size, dim)

        # compute nD Gaussian kernel as the product of 1d Gaussian kernels
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(ksize, dtype=torch.float32)
                                    for ksize in kernel_size])  # i-j order
        for ksize, sigm, mgrid in zip(kernel_size, sigmas, meshgrids):
            mean = (ksize - 1) / 2  # odd number kernel_size
            kernel *= 1 / (sigm * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / sigm) ** 2 / 2)

        # normalise to sum of 1
        self.kernel_norm_factor = kernel.sum()
        self.kernel = kernel / self.kernel_norm_factor   # size (kernel_size) * dim

        # Repeat the kernel on the out_channels dimension to the number of dimensions of data
        # each output channel of the kernel is used by each group in convolution when groups=dim
        # so each input channel is filtered by the same Gaussian kernel
        self.kernel = self.kernel.view(1, 1, *self.kernel.size())
        self.kernel = self.kernel.repeat(self.dim, *(1,) * (self.kernel.dim()-1))

        # set padding as half kernel size (valid)
        self.padding = [int(kernel_size[i]//2) for i in range(dim)]

        # get the convolution function of the right dimension
        self.conv_Nd_fn = getattr(F, f"conv{dim}d")

    def __call__(self, x):
        """
        Apply Gaussian smoothing using image convolution

        Args:
            x: (torch.Tensor) shape (N, ch, H, W) or (N, ch, H, W, D)
        Returns:
            output: (torch.Tensor) same shape as input, Gaussian filter smoothed
        """
        self.kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        output = self.conv_Nd_fn(x, self.kernel, padding=self.padding, groups=self.dim)
        return output


def synthesis_elastic_deformation(image,
                                  roi_mask,
                                  smooth_filter=None,
                                  cps=10,
                                  disp_max=1.,
                                  bbox_pad_ratio=0.2,
                                  device=torch.device('cpu')):
    """
    Synthesis elastic deformation in 2D and 3D.
    Randomly generate control points -> interpolation ->  Gaussian filter smoothing

    Args:
        image: (numpy.ndarray, shape (N, H, W) or (N, H, W, D))
        roi_mask:  (numpy.ndarray, shape (N, H, W) or (N, H, W, D))
        smooth_filter: (GaussianFilter instance) Gaussian filter for smoothing the interpolated DVF
        cps: (int or tuple/list) control point spacing
        disp_max: (float or tuple/list) maximum displacement
        bbox_pad_ratio: (float or tuple/list) ratio of padding of bounding box cropping (see utils.image.bbox_from_mask)
        device: (torch.device)

    Returns:
        image_deformed: (numpy.ndarray, shape same as input image)
        dvf: (numpy.ndarray, shape (N, 2, H, W) or (N, 3, H, W, D)) Synthesised dense Displacement Vector Field
    """
    dim = image.ndim - 1
    batch_size = image.shape[0]
    image_shape = image.shape[1:]

    # check & expand parameters to dimensions if needed
    cps = param_dim_setup(cps, dim)
    disp_max = param_dim_setup(disp_max, dim)
    bbox_pad_ratio = param_dim_setup(bbox_pad_ratio, dim)

    """Generate random elastic DVF """
    # randomly sample the control point parameters, weight by the scale factor
    cp_shape = tuple([image_shape[i] // cps[i] for i in range(dim)])
    cp_params = [np.random.uniform(-1, 1, cp_shape) * disp_max[i]
                 for i in range(dim)]
    cp_params = np.array(cp_params).astype(image.dtype)  # (dim, *(num_cp))

    # repeat along batch size dimension
    cp_params = np.tile(cp_params, (batch_size, *(1, ) * cp_params.ndim))

    # compute dense DVF by interpolate to image size (dim, *size)
    cp_params = torch.from_numpy(cp_params).to(device=device)

    inter_mode = "bilinear" if dim == 2 else "trilinear"
    dvf = F.interpolate(cp_params,
                        size=image_shape,
                        mode=inter_mode,
                        align_corners=False
                        )
    # apply smoothing filter if given
    if smooth_filter is not None:
        dvf = smooth_filter(dvf)  # (N, dim, *size)
    """"""

    # mask the DVF with ROI bounding box
    # todo: masking out with ROI mask bounding box is not necessary in synthesis
    mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask, pad_ratio=bbox_pad_ratio)
    dvf *= torch.from_numpy(mask_bbox_mask[:, np.newaxis, ...]).to(device=device)  # (N, dim, *size) * (N, 1, *size)

    # Deform image
    image = torch.from_numpy(image).unsqueeze(1).to(device=device)
    image_deformed = spatial_transform(image, dvf)  # (N, 1, *size)

    return image_deformed.squeeze(1).cpu().numpy(), dvf.cpu().numpy()
