""" Spatial differentiation functions """
import torch
import torch.nn.functional as F
import numpy as np


def finite_diff(x, mode="central", boundary="Neumann"):
    """Input shape (N, dim, *sizes), mode='foward', 'backward' or 'central'"""
    if mode == "central":
        x_diff_forward = finite_diff_oneside(x, direction="forward", boundary=boundary)
        x_diff_backward = finite_diff_oneside(x, direction="backward", boundary=boundary)
        return [(x_diff_fw + x_diff_bw) / 2
                for (x_diff_fw, x_diff_bw) in zip(x_diff_forward, x_diff_backward)]
    else: # "forward" or "backward"
        return finite_diff_oneside(x, direction=mode, boundary=boundary)


def finite_diff_oneside(x, direction="forward", boundary="Neumann"):
    """
    Calculate one-sided finite difference, works with 2D/3D Pytorch Tensor or Numpy Array.
    Forward difference: dx[i] = x[i+1] - x[i]
    Backward difference: dx[i] = x[i] - x[i-1]

    Args:
        x: (torch.Tensor or numpy.ndarray, size/shape (N, dim, *size)) The array to differentiate.
        direction: (string) Direction of the finite difference appoximation, "forward" or "backward".
        boundary: (string) Boundary condition, "Neumann" or "Dirichlet".

    Returns:
        x_diff
    """

    dim = x.ndim - 2
    sizes = x.shape[2:]

    # initialise finite difference list
    x_diff = []  # [x_dx, x_dy, x_dz]

    for i in range(dim):
        # configure padding of this dimension
        # (don't use []*dim as it just replicates pointers of the one array)
        paddings = [[0, 0] for j in range(dim)]

        if direction == "forward":
            # forward difference: pad after
            paddings[i][1] = 1
        else:
            # backward difference: pad before
            paddings[i][0] = 1


        if type(x) is np.ndarray:
            # add the first 2 dimensions for numpy
            paddings = [[0, 0], [0, 0]] + paddings

            # padding
            if boundary == "Neumann":
                # Neumann boundary condition
                x_pad = np.pad(x, paddings, mode='edge')
            elif boundary == "Dirichlet":
                # Dirichlet boundary condition
                x_pad = np.pad(x, paddings, mode='constant')
            else:
                raise ValueError("Boundary condition not recognised.")

            # slice and subtract
            x_diff += [x_pad.take(np.arange(1, sizes[i]+1), axis=i+2)
                       - x_pad.take(np.arange(0, sizes[i]), axis=i+2)]


        elif type(x) is torch.Tensor:
            # Pytorch uses last -> first dimension order
            paddings.reverse()
            # join sublists into a flat list
            paddings = tuple([p for ppair in paddings for p in ppair])

            # padding
            if boundary == "Neumann":
                # Neumann boundary condition
                x_pad = F.pad(x, paddings, mode='replicate')
            elif boundary == "Dirichlet":
                # Dirichlet boundary condition
                x_pad = F.pad(x, paddings, mode='constant')
            else:
                raise ValueError("Boundary condition not recognised.")

            # slice and subtract
            x_diff += [x_pad.index_select(i+2, torch.arange(1, sizes[i]+1))
                       - x_pad.index_select(i+2, torch.arange(0, sizes[i]))]

        else:
            raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")

    return x_diff
