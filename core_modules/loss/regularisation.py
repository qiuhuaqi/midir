""" Transformation regularisation losses """
import torch
from core_modules.loss.utils import finite_diff


def l2reg(x):
    """
    L2 regularisation of the spatial derivatives of displacement or velocity field

    Args:
        x: (torch.Tensor, size (N, dim, *sizes)) Dense vector field

    Returns:
        diffusion loss: (scalar) Diffusion (L2) regularisation loss
    """
    x_dxyz = finite_diff(x, mode="forward")
    return torch.cat(x_dxyz, dim=1).pow(2).sum(dim=1).mean()


def bending_energy(x):
    """
    Compute the Bending Energy regularisation loss (Rueckert et al., 1999)

    Args:
        x: (torch.Tensor, size (N, dim, *sizes)) Dense vector field

    Returns:
        BE: (scalar) Bending Energy loss
    """
    # 1st order derivatives
    dvf_d1 = finite_diff(x, mode="forward")

    # 2nd order derivatives
    dvf_d1 = torch.cat(dvf_d1, dim=1)
    dvf_d2 = finite_diff(dvf_d1, mode="forward")
    return torch.cat(dvf_d2, dim=1).pow(2).sum(dim=1).mean()


# TODO: Differentiable Jacobian constraint loss
# def compute_jacobian(x):
#     """ reference code from Chen"""
#     bsize, csize, height, width = x.size()
#     # padding
#     v = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()),
#                   3)
#     u = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()),
#                   2)
#
#     d_x = (torch.index_select(v, 3, torch.arange(2, width + 2).cuda())
#            - torch.index_select(v, 3, torch.arange(width).cuda())) / 2
#     d_y = (torch.index_select(u, 2, torch.arange(2, height + 2).cuda()) - torch.index_select(u, 2, torch.arange(
#         height).cuda())) / 2
#
#     J = (torch.index_select(d_x, 1, torch.tensor([0]).cuda())+1)*(torch.index_select(d_y, 1, torch.tensor([1]).cuda())+1) \
#         -torch.index_select(d_x, 1, torch.tensor([1]).cuda())*torch.index_select(d_y, 1, torch.tensor([0]).cuda())
#     return J
#
