# Transformation models #
from utils.misc import param_ndim_setup

from core_modules.mirtk_torch import cubic_bspline1d
from core_modules.mirtk_torch import conv1d


class DVFTransform(object):
    """ Displacement field model (dummy) """
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class CubicBSplineTransform(object):
    def __init__(self, ndim, img_size=192, cps=5):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.

        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            order: (int) B-spline order
        """
        self.ndim = ndim
        self.img_size = param_ndim_setup(img_size, self.ndim)
        self.stride = param_ndim_setup(cps, self.ndim)

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2 for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def __call__(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            dvf: (N, dim, *(img_sizes)) The dense Displacement Vector Field of the transformation
        """
        # separable 1d transposed convolution
        y = x
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            y = conv1d(y, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, y.shape[0]), slice(0, y.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        y = y[slicer]
        return y


class MultiResBSplineFFDTransform(object):
    def __init__(self, dim, img_size, lvls, cps):
        """
        Multi-resolution B-spline transformation

        Args:
            dim: (int) transformation model dimension
            img_size: (int or tuple) image size at original resolution
            lvls: (int) number of multi-resolution levels
            cps: (int) control point spacing at the original resolution
        """
        self.transforms = []
        for l in range(lvls):
            img_size_l = [imsz // (2 ** l) for imsz in img_size]
            cps_l = cps * (2 ** l)
            transform_l = CubicBSplineTransform(dim, img_size=img_size_l, cps=cps_l)
            self.transforms += [transform_l]

    def __call__(self, x):
        assert len(x) == len(self.transforms)
        dvfs = []
        for x_l, transform_l in zip(x, self.transforms):
            dvf_l = transform_l(x_l)
            dvfs += [dvf_l]
        return dvfs

