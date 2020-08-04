from torch import nn as nn

from model.losses import sim_loss, reg_loss
from model.networks import networks
from model.transformations import DVFTransform, BSplineFFDTransform


def get_network(hparams):
    """Configure network, return network instance"""
    network = getattr(networks, hparams.network.name)
    network = network(dim=hparams.data.dim, **hparams.network.net_config)
    return network


def get_transformation(hparams):
    """Configure transformation"""
    if hparams.transformation.type == "DVF":
        transformation = DVFTransform()

    elif hparams.transformation == "FFD":
        transformation = BSplineFFDTransform(dim=hparams.data.dim,
                                             img_size=hparams.data.crop_size,
                                             sigma=hparams.transformation.sigma)
    else:
        raise ValueError("Model: Transformation model not recognised")
    return transformation


def get_loss_fn(hparams):
    # similarity loss
    if hparams.loss.sim_loss == 'MSE':
        sim_loss_fn = nn.MSELoss()

    elif hparams.loss.sim_loss == 'NMI':
        sim_loss_fn = sim_loss.MILossGaussian(**hparams.loss.mi_cfg)
    else:
        raise ValueError(f'Similarity loss not recognised: {hparams.loss.sim_loss}.')

    # regularisation loss
    reg_loss_fn = getattr(reg_loss, hparams.loss.reg_loss)

    return sim_loss_fn, reg_loss_fn