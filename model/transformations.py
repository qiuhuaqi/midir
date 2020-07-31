import numpy as np
import torch
import torch.nn.functional as F
from model.window_func import cubic_bspline_torch
from utils.transformation import normalise_dvf
from utils.misc import param_dim_setup

""" 
Transformation models 
"""

class BSplineFFDTransform(object):
    def __init__(self,
                 dim,
                 img_size=(192, 192),
                 sigma=(5, 5),
                 order=3
                 ):
        """
        Compute dense Displacement Vector Field (DVF) of B-spline FFD transformation model
        from control point parameters, works with both 2D and 3D.
        B-spline kernel computed with recursive convolutions.

        Args:
            dim: (int) image dimension
            img_size: (int or tuple) size of the image, if int assume same size on all dimensions
            cpt_spacing: (int or tuple) number of points between adjacent control points
        """

        self.dim = dim

        # assume same for all dimensions if one integer is given
        self.img_size = param_dim_setup(img_size, self.dim)
        # self.cpt_spacing = param_dim_setup(cpt_spacing, self.dim)
        self.sigma = param_dim_setup(sigma, self.dim)

        self.conv_transposeNd_fn = getattr(F, f"conv_transpose{self.dim}d")

        self._set_kernel(order=order)

        self.stride = self.sigma
        self.padding = [int((ks-1)/2) for ks in self.kernel.size()[2:]]


    def _set_kernel(self, order=3):
        """
        Compute B-spline kernel of arbitrary order using recursive convolution
        Adapted from AirLab implementation:
        https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/utils/kernelFunction.py#L258

        `sigma` is the size of 0th order B-spline function,
        which determines the size of the n-th order B-spline kernel.
        Here, sigma is set using the convention that each control point controls +/- 1 cpt_spacing
        """
        kernel_ones = torch.ones(1, 1, *self.sigma)
        kernel = kernel_ones
        padding = np.array(self.sigma) - 1

        convNd_fn = getattr(F, f"conv{self.dim}d")

        for i in range(order):
            kernel = convNd_fn(kernel, kernel_ones, padding=(padding).tolist()) / np.prod(self.sigma)

        self.kernel = kernel.repeat(self.dim, 1, *(1,) * self.dim)  # (dim, 1, *(kernel_sizes))
        self.kernel = self.kernel


    def __call__(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) network output i.e. control point parameters

        Returns:
            dvf: (N, dim, *(img_sizes)) dense Displacement Vector Field of the transformation
        """
        self.kernel = self.kernel.to(dtype=x.dtype, device=x.device)

        # compute the DVF of the FFD transformation by transposed convolution 2D/3D
        dvf = self.conv_transposeNd_fn(x,
                                       weight=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding,
                                       groups=self.dim
                                       )

        # crop DVF to image size (centres aligned)
        for i in range(self.dim):
            assert dvf.size()[i + 2] >= self.img_size[i], \
                f"FFD output DVF size ({dvf.size()[i+2]}) is smaller than image size ({self.img_size[i]}) at dimension {i}"
            crop_start = dvf.size()[i + 2] // 2 - self.img_size[i] // 2
            dvf = dvf.narrow(i+2, crop_start, self.img_size[i])
        return dvf



class BSplineFFDTransformPoly(BSplineFFDTransform):
    def __init__(self,
                 dim,
                 img_size=(192, 192),
                 cpt_spacing=(8, 8)
                 ):
        """
        B-spline FFD transformation model
        with the kernel computed using the polynomial B-spline function formulation
        """
        super(BSplineFFDTransformPoly, self).__init__(dim=dim,
                                                      img_size=img_size,
                                                      cpt_spacing=cpt_spacing)
        self._set_kernel()
        self.strides = ([cps + 1 for cps in self.cpt_spacing])
        self.padding = [int((ks-1)/2) for ks in self.kernel.size()[2:]]

    def _set_kernel(self, order=3):
        # initialise the 1-d b-spline kernels (specific for cubic B-spline)
        kernels = [torch.arange(0, 4 * int(cps) + 1).float() for cps in self.cpt_spacing]

        # distance to kernel centre
        kernels = [k - len(k) // 2 for k in kernels]

        # cubic b-spline function
        kernels = [cubic_bspline_torch(k / self.cpt_spacing[i]) / self.cpt_spacing[i]
                   for i, k in enumerate(kernels)]

        # build the n-d conv kernel by outer-product of 1d filters using broadcasting
        kernel = kernels[0]
        for k in kernels[1:]:
            kernel = kernel.unsqueeze(-1) * k.unsqueeze(0)

        # extend shape to conv filter
        kernel = kernel.view(1, 1, *kernel.size())
        self.kernel = kernel.repeat(self.dim, 1, *(1,) * self.dim)  # (dim, 1, *(kernel_sizes))
        self.kernel = self.kernel.float()



class DVFTransform(object):
    """ (Dummy) Displacement Vector Field model """
    def __init__(self):
        pass

    def __call__(self, x):
        return x

""""""




"""
Spatial Transformer
"""

def spatial_transform(x, dvf, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

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
    deformed_meshgrid = [deformed_meshgrid[dim-1-i] for i in range(dim)]
    deformed_meshgrid = torch.stack(deformed_meshgrid, -1)  # (N, *size, dim)

    return F.grid_sample(x, deformed_meshgrid, mode=interp_mode, align_corners=False)
