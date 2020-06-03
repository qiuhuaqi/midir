import torch
import torch.nn.functional as F
from model.window_func import cubic_bspline_torch
from utils.transformation import normalise_dvf

""" 
Transformation models 
"""

class BSplineFFDTransform(object):
    def __init__(self,
                 dim=2,
                 img_size=(192, 192),
                 cpt_spacing=(8, 8)):
        """
        Compute dense Displacement Vector Field (DVF) of B-spline FFD transformation model
        from control point parameters, works with both 2D and 3D

        Args:
            dim: (int) image dimension
            img_size: (int or tuple) size of the image, if int assume same size on all dimensions
            cpt_spacing: (int or tuple) number of points between adjacent control points
        """

        self.dim = dim
        # assume same size for all dimensions if one integer is given
        if isinstance(img_size, int):
            self.img_size = (img_size,) * dim
        else:
            self.img_size = img_size

        if isinstance(cpt_spacing, int):
            self.cpt_spacing = (cpt_spacing,) * dim
        else:
            self.cpt_spacing = cpt_spacing

        self.strides = ([s + 1 for s in self.cpt_spacing])
        self._set_kernel()

    def _set_kernel(self):
        # initialise the 1-d b-spline kernels
        kernels = [torch.arange(0, 4 * int(cps) + 1).float() for cps in self.cpt_spacing]

        # distance to kernel centre
        kernels = [k - len(k) // 2 for k in kernels]

        # cubic b-spline function
        kernels = [(1 / self.cpt_spacing[i]) * cubic_bspline_torch(k / self.cpt_spacing[i])
                   for i, k in enumerate(kernels)]

        # build the n-d conv kernel by outer-product of 1d filters using broadcasting
        kernel = kernels[0]
        for k in kernels[1:]:
            kernel = kernel.unsqueeze(-1) * k.unsqueeze(0)

        # extend shape to conv filter
        kernel = kernel.view(1, 1, *kernel.size())
        self.kernel = kernel.repeat(self.dim, 1, *(1,) * self.dim)  # (dim, 1, *(kernel_sizes))
        self.kernel = self.kernel.float()

    def __call__(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) network output i.e. control point parameters

        Returns:
            dvf: (N, dim, *(img_sizes)) dense Displacement Vector Field of the transformation
        """
        self.kernel = self.kernel.to(device=x.device)

        # compute the DVF of the FFD transformation by transposed convolution 2D/3D
        conv_transposeNd_fn = getattr(F, f"conv_transpose{self.dim}d")
        dvf = conv_transposeNd_fn(x,
                               weight=self.kernel,
                               stride=self.strides,
                               groups=self.dim,
                               padding=self.cpt_spacing)

        # crop DVF to image size (centres aligned)
        for i in range(self.dim):
            assert dvf.size()[i + 2] >= self.img_size[i], \
                f"FFD DVF size is smaller than image at dimension {i}"

        crop_starts = [dvf.size()[i + 2] // 2 - self.img_size[i] // 2
                       for i in range(self.dim)]
        crop_ends = [crop_starts[i] + self.img_size[i]
                     for i in range(self.dim)]
        for i in range(self.dim):
            dvf = dvf.index_select(i + 2, torch.arange(crop_starts[i], crop_ends[i], device=x.device))
        return dvf


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
    Spatially transform an image by sampling at transformed locations
    # todo: add affine mode

    Args:
        x: (Tensor float, shape (N, Ch, H, W) or (N, ch, H, W, D)) image to be spatially transformed
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
    mesh_grid = torch.meshgrid([torch.linspace(-1, 1, size[i], dtype=dvf.dtype) for i in range(dim)])
    mesh_grid = [mesh_grid[i].requires_grad_(False).to(device=dvf.device) for i in range(dim)]

    # apply displacements to each direction (N, *size)
    deformed_meshgrid = [mesh_grid[i] + dvf[:, i, ...] for i in range(dim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    deformed_meshgrid = [deformed_meshgrid[dim-1-i] for i in range(dim)]
    deformed_meshgrid = torch.stack(deformed_meshgrid, -1)  # (N, *size, dim)

    return F.grid_sample(x, deformed_meshgrid, mode=interp_mode)
