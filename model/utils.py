import random

import numpy as np
import torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Any, Dict

from core_modules.network.nets import UNet, MultiResUNet, CubicBSplineNet
from core_modules.transform.transformations import DenseTransform, CubicBSplineFFDTransform
from core_modules.loss import similarity, regularisation
from core_modules.loss.utils import MultiResLoss


def get_network(hparams):
    """Configure network"""
    if hparams.network.name == "unet":
        network = UNet(ndim=hparams.data.ndim,
                       **hparams.network.net_config)

    elif hparams.network.name == "mulunet":
        network = MultiResUNet(ndim=hparams.data.ndim,
                               ml_lvls=hparams.meta.ml_lvls,
                               **hparams.network.net_config)

    elif hparams.network.name == "bspline_net":
        network = CubicBSplineNet(ndim=hparams.data.ndim,
                                  img_size=hparams.data.crop_size,
                                  cps=hparams.transformation.config.cps,
                                  **hparams.network.net_config)

    else:
        raise ValueError("Model config parsing: Network not recognised")
    return network


def get_transformation(hparams):
    """Configure transformation"""
    if hparams.transformation.type == "DENSE":
        transformation = DenseTransform(lvls=hparams.meta.ml_lvls,
                                        **hparams.transformation.config
                                        )

    elif hparams.transformation.type == "BSPLINE":
        transformation = CubicBSplineFFDTransform(ndim=hparams.data.ndim,
                                                  lvls=hparams.meta.ml_lvls,
                                                  img_size=hparams.data.crop_size,
                                                  **hparams.transformation.config
                                                  )
    else:
        raise ValueError("Model config parsing: Transformation model not recognised")
    return transformation


def get_loss_fn(hparams):
    # similarity loss
    if hparams.loss.sim_loss == 'MSE':
        sim_loss_fn = nn.MSELoss()

    elif hparams.loss.sim_loss == 'LNCC':
        sim_loss_fn = similarity.LNCCLoss(hparams.loss.window_size)

    elif hparams.loss.sim_loss == 'NMI':
        sim_loss_fn = similarity.MILossGaussian(**hparams.loss.mi_cfg)

    else:
        raise ValueError(f'Similarity loss not recognised: {hparams.loss.sim_loss}.')

    # regularisation loss
    reg_loss_fn = getattr(regularisation, hparams.loss.reg_loss)

    # multi-resolution loss function
    loss_fn = MultiResLoss(sim_loss_fn,
                           hparams.loss.sim_loss,
                           reg_loss_fn,
                           hparams.loss.reg_loss,
                           reg_weight=hparams.loss.reg_weight,
                           ml_lvls=hparams.meta.ml_lvls,
                           ml_weights=hparams.loss.ml_weights)
    return loss_fn


def worker_init_fn(worker_id):
    """ Callback function passed to DataLoader to initialise the workers """
    # Randomly seed the workers
    random_seed = random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        # looks for `hparams` and `hparam_metrics` in `pl_module`
        pl_module.logger.log_metrics(pl_module.hparam_metrics,
                                     step=pl_module.global_step)
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
        }
