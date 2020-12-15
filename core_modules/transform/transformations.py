# Transformation models #
from utils.misc import param_ndim_setup

from core_modules.transform.utils import svf_exp, cubic_bspline1d, conv1d


class _Transform(object):
    """ Transformation base class """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        self.svf = svf
        self.svf_steps = svf_steps
        self.svf_scale = svf_scale

    def compute_flow(self, x):
        raise NotImplementedError

    def __call__(self, x):
        flow = self.compute_flow(x)
        if self.svf:
            disp = svf_exp(flow,
                           scale=self.svf_scale,
                           steps=self.svf_steps)
            return flow, disp
        else:
            disp = flow
            return disp


class _DenseTransform(_Transform):
    """ Dense field transformation """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        super(_DenseTransform, self).__init__(svf=svf,
                                              svf_steps=svf_steps,
                                              svf_scale=svf_scale)

    def compute_flow(self, x):
        return x


class _CubicBSplineTransform(_Transform):
    def __init__(self,
                 ndim,
                 img_size=192,
                 cps=5,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.

        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            svf: (bool) stationary velocity field formulation if True
        """
        super(_CubicBSplineTransform, self).__init__(svf=svf,
                                                     svf_steps=svf_steps,
                                                     svf_scale=svf_scale)
        self.ndim = ndim
        self.img_size = param_ndim_setup(img_size, self.ndim)
        self.stride = param_ndim_setup(cps, self.ndim)

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2
                        for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def compute_flow(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            y: (N, dim, *(img_sizes)) The dense flow field of the transformation
        """
        # separable 1d transposed convolution
        flow = x
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]
        return flow


class _MultiResTransform(object):
    """ Multi-resolution transformation base class """
    def __init__(self,
                 lvls=1,
                 svf=False):
        self.lvls = lvls
        self.svf = svf
        self.transforms = list()  # needs to implement

    def __call__(self, x):
        assert len(x) == self.lvls
        assert len(self.transforms) == self.lvls

        disps = list()
        if self.svf:
            flows = list()
            for x_l, transform in zip(x, self.transforms):
                flow, disp = transform(x_l)
                flows.append(flow)
                disps.append(disp)
            return flows, disps
        else:
            for x_l, transform in zip(x, self.transforms):
                disp = transform(x_l)
                disps.append(disp)
            return disps


class DenseTransform(_MultiResTransform):
    """ Multi-resolution dense field model """
    def __init__(self,
                 lvls=1,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        super(DenseTransform, self).__init__(lvls=lvls, svf=svf)
        self.transforms = [_DenseTransform(svf=svf,
                                           svf_steps=svf_steps,
                                           svf_scale=svf_scale)
                           for _ in range(self.lvls)]


class CubicBSplineFFDTransform(_MultiResTransform):
    def __init__(self,
                 ndim,
                 img_size,
                 cps,
                 lvls=1,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        """
        Multi-resolution B-spline transformation

        Args:
            ndim: (int) transformation model dimension
            lvls: (int) number of multi-resolution levels
            img_size: (int or tuple) image size at original resolution
            cps: (int) control point spacing at the original resolution
        """
        super(CubicBSplineFFDTransform, self).__init__(lvls=lvls,
                                                       svf=svf)
        for l in range(self.lvls):
            img_size_l = [imsz // (2 ** l) for imsz in img_size]
            # note: the control point spacing is effectively doubled
            #  as the image size halves
            transform = _CubicBSplineTransform(ndim,
                                               img_size=img_size_l,
                                               cps=cps,
                                               svf=svf,
                                               svf_steps=svf_steps,
                                               svf_scale=svf_scale)
            self.transforms.append(transform)
        self.transforms.reverse()
