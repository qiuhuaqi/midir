import torch
import torch.nn.functional as F
from model.window_func import cubic_bspline_torch

class BSplineFFDTransform(object):
    def __init__(self,
                 crop_size=192,
                 cps=8):
        """
        (i,j) refer to points on spatially on each DVF dimensions,
        where (x, y) refer to the dimensions of DVF

        Args:
            cps: control point spacing
        """
        self.crop_size = crop_size
        self.cps = cps

        # design the kernel
        # kernel_size = (2 * cps + 1) * 2 + 1  # each point is affected by 4x4 neighbouring control points
        # # control point distance from kernel centre
        # cp_i = torch.arange(0, kernel_size) - cps * 2 - 1
        # cp_j = torch.arange(0, kernel_size) - cps * 2 - 1

        # or this?
        kernel_size = 2 * cps + 1
        cp_i = torch.arange(0, kernel_size) - cps - 1
        cp_j = torch.arange(0, kernel_size) - cps - 1

        # expand to 2D image
        kernel_i = torch.ones((kernel_size, kernel_size)) * cp_i.unsqueeze(1)
        kernel_j = torch.ones((kernel_size, kernel_size)) * cp_j.unsqueeze(0)

        # pass through B-spline kernel
        bspline_kernel_i = cubic_bspline_torch(kernel_i / cps) * (1 / cps)
        bspline_kernel_j = cubic_bspline_torch(kernel_j / cps) * (1 / cps)

        # compute 2d filter from seperable 1d filters
        kernel = bspline_kernel_i * bspline_kernel_j
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
        self.kernel = self.kernel.repeat(2, 1, 1, 1).float()  # (2, 1, kernel_size, kernel_size)

    def __call__(self, x):
        """
        Args:
            x: (N, 2, Kh, Kw) network output i.e. control point parameters

        Returns:
            dvf: (N, 2, H, W)
            warped_source: (N, 1, H, W)
        """
        # compute DVF from B-spline control point parameters
        self.kernel = self.kernel.to(device=x.device)
        dvf = F.conv_transpose2d(x, weight=self.kernel, stride=self.cps + 1, groups=2)

        # crop out image size DVF
        # todo: only works with SiameseFCN,rasm symmetrical image now
        crop_start = dvf.size()[-1]//2 - self.crop_size//2
        crop_end =  crop_start + self.crop_size
        dvf = dvf[:, :, crop_start: crop_end, crop_start:crop_end]
        return dvf


class DVFTransform(object):
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
