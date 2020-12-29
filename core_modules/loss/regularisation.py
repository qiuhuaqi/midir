""" Transformation regularisation losses """
import math
import torch
from core_modules.loss.utils import finite_diff


def l2reg_loss(u):
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def bending_energy_loss(u):
    derives = []
    ndim = u.size()[1]
    # 1st order
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    # 2nd order
    derives2 = []
    for i in range(ndim):
        derives2 += [finite_diff(derives[i], dim=i)]  # du2xx, du2yy, (du2zz)
    derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=1)]  # du2dxy
    if ndim == 3:
        derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=2)]  # du2dxz
        derives2 += [math.sqrt(2) * finite_diff(derives[1], dim=2)]  # du2dyz

    assert len(derives2) == 2 * ndim
    loss = torch.cat(derives2, dim=1).pow(2).sum(dim=1).mean()
    return loss



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
