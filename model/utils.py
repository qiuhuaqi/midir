import os
import shutil

import torch

from model.network.single_res import UNet, FFDNet
from model.network.multi_res import mlUNet
from model.transformations import DVFTransform, BSplineFFDTransform
import torch.nn as nn
from model.loss import sim_loss, reg_loss
from model.loss.mul_loss import mulLoss


def get_network(hparams):
    """Configure network"""
    if hparams.network.name == "UNet":
        network = UNet(dim=hparams.data.dim,
                       **hparams.network.net_config)

    elif hparams.network.name == "FFDNet":
        network = FFDNet(dim=hparams.data.dim,
                         img_size=hparams.data.crop_size,
                         cpt_spacing=hparams.transformation.sigma,
                         **hparams.network.net_config)

    elif hparams.network.name == "mulUNet":
        network = mlUNet(dim=hparams.data.dim,
                         ml_lvls=hparams.meta.ml_lvls,
                         **hparams.network.net_config)

    elif hparams.network.name == "mlFFDNet":
        raise NotImplementedError

    else:
        raise ValueError("Model config parsing: Network not recognised")
    return network


def get_transformation(hparams):
    """Configure transformation"""
    if hparams.transformation.type == "DVF":
        transformation = DVFTransform()

    elif hparams.transformation.type == "FFD":
        transformation = BSplineFFDTransform(dim=hparams.data.dim,
                                             img_size=hparams.data.crop_size,
                                             sigma=hparams.transformation.sigma)
    else:
        raise ValueError("Model config parsing: Transformation model not recognised")
    return transformation


def get_loss_fn(hparams):
    # similarity loss
    if hparams.loss.sim_loss == 'MSE':
        sim_loss_fn = nn.MSELoss()

    elif hparams.loss.sim_loss == 'LNCC':
        sim_loss_fn = sim_loss.LNCCLoss(hparams.loss.window_size)

    elif hparams.loss.sim_loss == 'NMI':
        sim_loss_fn = sim_loss.MILossGaussian(**hparams.loss.mi_cfg)

    else:
        raise ValueError(f'Similarity loss not recognised: {hparams.loss.sim_loss}.')

    # regularisation loss
    reg_loss_fn = getattr(reg_loss, hparams.loss.reg_loss)

    # multi-resolution loss function
    loss_fn = mulLoss(sim_loss_fn,
                      hparams.loss.sim_loss,
                      reg_loss_fn,
                      hparams.loss.reg_loss,
                      reg_weight=hparams.loss.reg_weight,
                      ml_lvls=hparams.meta.ml_lvls,
                      ml_weights=hparams.loss.ml_weights)
    return loss_fn


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        if state['epoch'] == 1:
            print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
