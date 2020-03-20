import torch
import torch.nn.functional as F
from model.window_func import cubic_bspline_torch
from model.submodules import spatial_transform


class BSplineFFDTransform():
    def __init__(self, cps=8):
        """
        (i,j) refer to points on spatially on each DVF dimensions,
        where (x, y) refer to the dimensions of DVF

        Args:
            cps: control point spacing
        """
        self.cps = cps

        # design the kernel
        kernel_size = (2 * cps + 1) * 2 + 1  # each point is affected by 4x4 neighbouring control points

        # control point distance from kernel centre
        cp_i = torch.arange(0, kernel_size) - cps * 2 - 1
        cp_j = torch.arange(0, kernel_size) - cps * 2 - 1
        # expand to 2D image
        kernel_i = torch.ones((kernel_size, kernel_size)) * cp_i.unsqueeze(1)
        kernel_j = torch.ones((kernel_size, kernel_size)) * cp_j.unsqueeze(0)

        # pass through B-spline kernel
        bspline_kernel_i = cubic_bspline_torch(kernel_i / cps) * (1 / cps)
        bspline_kernel_j = cubic_bspline_torch(kernel_j / cps) * (1 / cps)
        # bspline_kernel_i = cubic_bspline_torch(kernel_i / cps)
        # bspline_kernel_j = cubic_bspline_torch(kernel_j / cps)  # proper scalling results in very small initial values

        # product of filters of the 2 dimensions
        kernel = bspline_kernel_i * bspline_kernel_j
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
        self.kernel = self.kernel.repeat(2, 1, 1, 1).float()  # (2, 1, kernel_size, kernel_size)

    def __call__(self, x, source):
        """
        Args:
            x: (N, 2, Kh, Kw) network output used as control point parameters
            source: (N, 1, H, W) source image

        Returns:
            dvf: (N, 2, H, W)
            warped_source: (N, 1, H, W)
        """
        # compute DVF from B-spline control point parameters
        self.kernel = self.kernel.to(device=source.device)
        dvf = F.conv_transpose2d(x, weight=self.kernel, stride=self.cps + 1, groups=2)

        ## debug: see kernel value
        # print(self.kernel.mean(), self.kernel.max())

        # todo: only works with symmetrical image now
        crop_start = dvf.size()[-1]//2 - source.size()[-1]//2
        crop_end =  crop_start + source.shape[-1]
        dvf = dvf[:, :, crop_start: crop_end, crop_start:crop_end]
        assert dvf.size()[-2:] == source.size()[-2:], "Cropped FFD-DVF has different size as the image"

        # apply spatial transformation
        warped_source = spatial_transform(source, dvf)
        return dvf, warped_source


class OpticalFlowTransform():
    def __init__(self):
        pass
    def __call__(self, x, source):
        """
        Optical flow transformation model

        Args:
            x: (N, 2, H, W) network output, should be full image size
            source: (N, 1, H, W) source image

        Returns:
            dvf: (N, 2, H, W)
            warped_source: (N, 1, H, W)

        """
        # apply spatial transformation
        dvf = x
        warped_source = spatial_transform(source, dvf)
        return dvf, warped_source

