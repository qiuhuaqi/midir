import torch
import torch.nn.functional as F
from model.window_func import cubic_bspline_torch

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
            img_sizes: sizes of the image of all dimensions
            cpt_spacings: control point spacings (number of points between adjacent control points)
        """

        self.dim = dim
        if len(cpt_spacing) == dim:
            self.cpt_spacing = cpt_spacing
        else:
            self.cpt_spacing = (cpt_spacing,) * dim

        if len(img_size) == dim:
            self.img_size = img_size
        else:
            self.img_size = (img_size,) * dim

        self._set_kernel()
        self.strides = ([s + 1 for s in self.cpt_spacing])

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
        if self.dim == 2:
            dvf = F.conv_transpose2d(x, weight=self.kernel, stride=self.strides, groups=self.dim,
                                     padding=self.cpt_spacing)
        elif self.dim == 3:
            dvf = F.conv_transpose3d(x, weight=self.kernel, stride=self.strides, groups=self.dim,
                                     padding=self.cpt_spacing)
        else:
            raise ValueError("FFD transformation model dimension unknown.")

        # crop DVF to image size (centres aligned)
        for i in range(self.dim):
            assert dvf.size()[i + 2] >= self.img_size[i], \
                f"FFD DVF size is smaller than image at dimension {i}"
        crop_starts = [dvf.size()[i + 2] // 2 - self.img_size[i] // 2
                       for i in range(self.dim)]
        crop_ends = [crop_starts[i] + self.img_size[i]
                     for i in range(self.dim)]
        for i in range(self.dim):
            dvf = dvf.index_select(i + 2, torch.arange(crop_starts[i], crop_ends[i]))

        return dvf


# class BSplineFFDTransform(object):
#     def __init__(self,
#                  crop_size=192,
#                  cps=8):
#         """
#         (i,j) refer to points on spatially on each DVF dimensions,
#         where (x, y) refer to the dimensions of DVF
#
#         Args:
#             cps: control point spacing
#         """
#         self.crop_size = crop_size
#         self.cps = cps
#
#         # design the kernel
#         # kernel_size = (2 * cps + 1) * 2 + 1  # each point is affected by 4x4 neighbouring control points
#         # # control point distance from kernel centre
#         # cp_i = torch.arange(0, kernel_size) - cps * 2 - 1
#         # cp_j = torch.arange(0, kernel_size) - cps * 2 - 1
#
#         # or 2x2 neighbouring control points as in Dirk Locxx's thesis?
#         kernel_size = 2 * cps + 1
#         cp_i = torch.arange(0, kernel_size) - cps - 1
#         cp_j = torch.arange(0, kernel_size) - cps - 1
#
#         # expand to 2D image
#         kernel_i = torch.ones((kernel_size, kernel_size)) * cp_i.unsqueeze(1)
#         kernel_j = torch.ones((kernel_size, kernel_size)) * cp_j.unsqueeze(0)
#
#         # pass through B-spline kernel
#         bspline_kernel_i = cubic_bspline_torch(kernel_i / cps) * (1 / cps)
#         bspline_kernel_j = cubic_bspline_torch(kernel_j / cps) * (1 / cps)
#
#         # compute 2d filter from seperable 1d filters
#         kernel = bspline_kernel_i * bspline_kernel_j
#         self.kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
#         self.kernel = self.kernel.repeat(2, 1, 1, 1).float()  # (2, 1, kernel_size, kernel_size)
#
#     def __call__(self, x):
#         """
#         Args:
#             x: (N, 2, Kh, Kw) network output i.e. control point parameters
#
#         Returns:
#             dvf: (N, 2, H, W)
#             warped_source: (N, 1, H, W)
#         """
#         # compute DVF from B-spline control point parameters
#         self.kernel = self.kernel.to(device=x.device)
#         dvf = F.conv_transpose2d(x, weight=self.kernel, stride=self.cps + 1, groups=2)
#
#         # crop out image size DVF
#         # todo: only works with SiameseFCN,rasm symmetrical image now
#         crop_start = dvf.size()[-1]//2 - self.crop_size//2
#         crop_end =  crop_start + self.crop_size
#         dvf = dvf[:, :, crop_start: crop_end, crop_start:crop_end]
#         return dvf


class DVFTransform(object):
    """ Dummy Displacement Vector Field model """

    def __init__(self):
        pass

    def __call__(self, x):
        return x


def spatial_transform(x, dvf, mode="bilinear"):
    """
    Spatially transform an image by sampling at coordinates of the deformed mesh grid.

    Args:
        x: source image, Tensor of shape (N, Ch, H, W)
        dvf: (Tensor, Nx2xHxW) displacement vector field from target to source, in [-1,1] coordinate
        mode: (striis) method of interpolation

    Returns:
        source image deformed using the deformation flow field,
        Tensor of the same shape as source image

    """

    # generate standard mesh grid
    h, w = x.size()[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])

    grid_h = grid_h.requires_grad_(False).to(device=x.device)
    grid_w = grid_w.requires_grad_(False).to(device=x.device)

    # (H,W) + (N, H, W) add by broadcasting
    new_grid_h = grid_h + dvf[:, 0, ...]
    new_grid_w = grid_w + dvf[:, 1, ...]

    # using x-y (column_num, row_num) order
    deformed_grid = torch.stack((new_grid_w, new_grid_h), 3)  # shape (N, H, W, 2)
    deformed_image = F.grid_sample(x.type_as(deformed_grid), deformed_grid,
                                   mode=mode, padding_mode="border", align_corners=True)

    return deformed_image
