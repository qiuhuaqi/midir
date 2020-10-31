"""Window functions used in Parzen window probability density estimation"""
import numpy as np
import torch
import torch.nn as nn

"""
Rectangle window functions
"""
def rect_window_numpy(x):
    """
    Rectangle window function centred at 0 and of width 1
    which operates on input array element-wisely inplace.

    Args:
        x: input numpy array

    Returns:
        x: window function response of the input array
    """
    x1mask = np.logical_and((x >= -0.5), (x < 0.5))
    x0mask = np.logical_or((x < -0.5), (x >= 0.5))

    x[x1mask] = 1
    x[x0mask] = 0
    return x


def rect_window_torch(x):
    """
    Rectangle window function centred at 0 and of width 1
    which operates on input tensor element-wisely inplace.

    Args:
        x: input tensor

    Returns:
        x: window function response of the input tensor
    """
    x1mask = (x > -0.5) * (x < 0.5)
    x0mask = ((x <= -0.5) + (x >= 0.5)) > 0

    x[x1mask] = 1
    x[x0mask] = 0
    return x


"""
Cubic B-spline window functions
"""

def cubic_bspline_numpy(x):
    """
    Cubic B-spline function in piece-wise polynomial
    which operates on input array element-wisely inplace.
    Centred at 0.

    Args:
        x:

    Returns:
        x:
    """
    x = np.absolute(x.astype("float"))

    # compute the masks
    mask_cond1 = (x < 1)
    mask_cond2 = (x >= 1) * (x < 2)
    mask_cond3 = (x >= 2)

    # |x| < 1
    x[mask_cond1] = 2 / 3 - np.square(x[mask_cond1]) + 0.5 * np.power(x[mask_cond1], 3)

    # 1 <= |x| < 2
    x[mask_cond2] = - (1 / 6) * np.power(x[mask_cond2] - 2, 3)

    # |x| >= 2
    x[mask_cond3] = 0
    return x


def grad_cubic_bspline_numpy(x):
    """
    Calculate the gradient of cubic B-spline function element-wisely

    Args:
        x:

    Returns:
        x:
    """
    xabs = np.absolute(x.astype("float"))
    grad_x = np.zeros_like(xabs)

    # compute the masks
    mask_cond1 = (xabs < 1)
    mask_cond2 = (xabs >= 1) * (xabs < 2)
    mask_cond3 = (xabs >= 2)

    # |x| < 1
    grad_x[mask_cond1] = -2 * x[mask_cond1] + 1.5 * xabs[mask_cond1] * x[mask_cond1]

    # 1 <= |x| < 2
    grad_x[mask_cond2] = - 0.5 * ((xabs[mask_cond2] - 2) ** 2) * (x[mask_cond2] / xabs[mask_cond2])

    # |x| >= 2, zeros as it is

    return grad_x


def cubic_bspline_torch(x):
    """
    Cubic B-spline function in piece-wise polynomial
    which operates on input tensor element-wisely inplace.

    Args:
        x: input tensor

    Returns:
        op: cubic B-spline function response of the input tensor
    """
    # find absolute value
    op = torch.abs(x)

    # compute the masks
    mask_cond1 = (op < 1)
    mask_cond2 = (op >= 1) * (op < 2)
    mask_cond3 = (op >= 2)

    # |x| < 1
    op[mask_cond1] = - op[mask_cond1] ** 2 + 0.5 * torch.pow(op[mask_cond1], 3) + 2/3

    # 1 <= |x| < 2
    op[mask_cond2] = - (1/6) * torch.pow(op[mask_cond2] - 2, 3)

    # |x| >= 2
    op[mask_cond3] = 0
    return op




class CubicBsplineTorch(torch.autograd.Function):
    """Implement custom forward and backward for the cubic B-spline module"""
    @staticmethod
    def forward(ctx, x):
        """
        Cubic B-spline function in piece-wise polynomial
        which operates on input tensor element-wisely inplace.

        Args:
            x: input tensor

        Returns:
            op: cubic B-spline function response of the input tensor
        """
        # find absolute value
        op = torch.abs(x)
        # todo: should ctx.mark_dirty() be used on op since in-place operations?

        # compute the masks
        mask_cond1 = (op < 1)
        mask_cond2 = (op >= 1) * (op < 2)
        mask_cond3 = (op >= 2)

        # save input and the masks to compute gradient
        ctx.save_for_backward(x, mask_cond1, mask_cond2, mask_cond3)

        # |x| < 1
        op[mask_cond1] = - op[mask_cond1] ** 2 + 0.5 * torch.pow(op[mask_cond1], 3) + 2/3

        # 1 <= |x| < 2
        op[mask_cond2] = - (1/6) * torch.pow(op[mask_cond2] - 2, 3)

        # |x| >= 2
        op[mask_cond3] = 0
        return op

    @staticmethod
    def backward(ctx, grad_output):
        x, mask_cond1, mask_cond2, mask_cond3 = ctx.saved_tensors

        # |x| < 1, db/dx = -2*x + 3/2*|x|*x
        grad_output[mask_cond1] = grad_output[mask_cond1] \
                                  * (- 2 * x[mask_cond1] + 1.5 * torch.abs(x[mask_cond1]) * x[mask_cond1])

        # 1 <= |x| < 2, db/dx = -1/2 * (|x| - 2)^2 * (x/|x|)
        grad_output[mask_cond2] = grad_output[mask_cond2] \
                        * (- 0.5 * ((torch.abs(x[mask_cond2]) - 2) ** 2) * (x[mask_cond2] / torch.abs(x[mask_cond2])))

        # |x| >= 2
        grad_output[mask_cond3] = 0

        # todo: consider create a Tensor copy of `grad_output` instead of in-place
        return grad_output
