import unittest
import torch
import numpy as np
import sys

sys.path.append('../')

from optoth.nabla import Nabla
from optoth.nabla2 import Nabla2

class TestDiffusionLoss(unittest.TestCase):
    def test(self):
        H = 64
        W = 64

        th_x = torch.randn(1, 1, H, W).cuda()
        th_loss = archive.legacy_model.losses.reg_loss.l2reg(th_x)

        # optoth - cannot deal with [nBatch, nCh, H, W] yet
        nabla = Nabla(dim=2).cuda()
        th_y = nabla(th_x[0,0])
        th_loss_2 = (th_y[0].pow(2) + \
                     th_y[1].pow(2)).mean()
        
        assert np.allclose(th_loss.cpu().numpy(), th_loss_2.cpu().numpy())

class TestBendingLoss(unittest.TestCase):
    def test(self):
        H = 64
        W = 64

        th_x = torch.randn(1, 1, H, W).cuda()
        th_loss = archive.legacy_model.losses.reg_loss.bending_energy(th_x)

        # optoth - cannot deal with [nBatch, nCh, H, W] yet
        nabla  = Nabla(dim=2).cuda()
        nabla2 = Nabla2(dim=2).cuda()
        th_y = nabla2(nabla(th_x[0,0]))
        th_loss_2 = (th_y[0].pow(2) + \
                     th_y[1].pow(2) + \
                     th_y[2].pow(2) + \
                     th_y[3].pow(2)).mean()

        assert np.allclose(th_loss.cpu().numpy(), th_loss_2.cpu().numpy())
