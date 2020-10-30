import torch.nn as nn

from model.network.nets import UNet, MultiResUNet, CubicBSplineNet
from model.transformations import DVFTransform, MultiResBSplineFFDTransform
from model.loss import sim_loss, reg_loss
from model.loss.mul_loss import MultiResLoss


def get_network(hparams):
    """Configure network"""
    if hparams.network.name == "unet":
        network = UNet(ndim=hparams.data.ndim,
                       **hparams.network.net_config)

    elif hparams.network.name == "mulunet":
        network = MultiResUNet(ndim=hparams.data.ndim,
                               ml_lvls=hparams.meta.ml_lvls,
                               **hparams.network.net_config)

    # elif hparams.network.name == "ffdnet":
    #     network = FFDNet(dim=hparams.data.ndim,
    #                      img_size=hparams.data.crop_size,
    #                      cpt_spacing=hparams.transformation.sigma,
    #                      **hparams.network.net_config)

    elif hparams.network.name == "bspline_net":
        network = CubicBSplineNet(ndim=hparams.data.ndim,
                                  img_size=hparams.data.crop_size,
                                  cps=hparams.transformation.cps,
                                  **hparams.network.net_config)

    else:
        raise ValueError("Model config parsing: Network not recognised")
    return network


def get_transformation(hparams):
    """Configure transformation"""
    if hparams.transformation.type == "DVF":
        transformation = DVFTransform()

    elif hparams.transformation.type == "FFD":
        transformation = MultiResBSplineFFDTransform(dim=hparams.data.ndim,
                                                     img_size=hparams.data.crop_size,
                                                     lvls=hparams.meta.ml_lvls,
                                                     cps=hparams.transformation.cps)
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
    loss_fn = MultiResLoss(sim_loss_fn,
                           hparams.loss.sim_loss,
                           reg_loss_fn,
                           hparams.loss.reg_loss,
                           reg_weight=hparams.loss.reg_weight,
                           ml_lvls=hparams.meta.ml_lvls,
                           ml_weights=hparams.loss.ml_weights)
    return loss_fn
