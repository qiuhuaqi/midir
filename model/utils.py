import os
import random
from typing import Any, Dict
import numpy as np
import torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from data.datasets import BrainMRInterSubj3D, CardiacMR2D
from model.network import UNet, MultiResUNet, CubicBSplineNet
from model.transformation import DenseTransform, CubicBSplineFFDTransform
from model import loss
from model.loss import MultiResLoss


def get_network(hparams):
    """Configure network"""
    if hparams.network.type == "unet":
        network = UNet(ndim=hparams.data.ndim,
                       **hparams.network.net_config)

    elif hparams.network.type == "mulunet":
        network = MultiResUNet(ndim=hparams.data.ndim,
                               ml_lvls=hparams.meta.ml_lvls,
                               **hparams.network.net_config)

    elif hparams.network.type == "bspline_net":
        network = CubicBSplineNet(ndim=hparams.data.ndim,
                                  img_size=hparams.data.crop_size,
                                  cps=hparams.transformation.config.cps,
                                  **hparams.network.net_config)

    else:
        raise ValueError(f"Network config ({hparams.network.name}) not recognised.")
    return network


def get_transformation(hparams):
    """Configure transformation"""
    if hparams.transformation.type == "dense":
        transformation = DenseTransform(lvls=hparams.meta.ml_lvls,
                                        **hparams.transformation.config
                                        )

    elif hparams.transformation.type == "bspline":
        transformation = CubicBSplineFFDTransform(ndim=hparams.data.ndim,
                                                  lvls=hparams.meta.ml_lvls,
                                                  img_size=hparams.data.crop_size,
                                                  **hparams.transformation.config
                                                  )
    else:
        raise ValueError(f"Transformation config ({hparams.transformation.type}) not recognised.")
    return transformation


def get_loss_fn(hparams):
    # similarity loss
    if hparams.loss.sim_loss == 'mse':
        sim_loss_fn = nn.MSELoss()

    elif hparams.loss.sim_loss == 'lncc':
        sim_loss_fn = loss.LNCCLoss(hparams.loss.window_size)

    elif hparams.loss.sim_loss == 'nmi':
        sim_loss_fn = loss.MILossGaussian(**hparams.loss.mi_cfg)

    else:
        raise ValueError(f'Similarity loss config ({hparams.loss.sim_loss}) not recognised.')

    # regularisation loss
    reg_loss_fn = getattr(loss, hparams.loss.reg_loss)

    # multi-resolution loss function
    loss_fn = MultiResLoss(sim_loss_fn,
                           hparams.loss.sim_loss,
                           reg_loss_fn,
                           hparams.loss.reg_loss,
                           reg_weight=hparams.loss.reg_weight,
                           ml_lvls=hparams.meta.ml_lvls,
                           ml_weights=hparams.loss.ml_weights)
    return loss_fn


def get_datasets(hparams):
    assert os.path.exists(hparams.data.train_path), \
        f"Training data path does not exist: {hparams.data.train_path}"
    assert os.path.exists(hparams.data.val_path), \
        f"Validation data path does not exist: {hparams.data.val_path}"

    if hparams.data.name == 'brain_camcan':
        train_dataset = BrainMRInterSubj3D(hparams.data.train_path,
                                           hparams.data.crop_size,
                                           modality=hparams.data.modality,
                                           atlas_path=hparams.data.atlas_path)

        val_dataset = BrainMRInterSubj3D(hparams.data.val_path,
                                         hparams.data.crop_size,
                                         evaluate=True,
                                         modality=hparams.data.modality,
                                         atlas_path=hparams.data.atlas_path)

    elif hparams.data.name == 'cardiac_ukbb':
        train_dataset = CardiacMR2D(hparams.data.train_path,
                                    crop_size=hparams.data.crop_size,
                                    slice_range=hparams.data.slice_range,
                                    slicing=hparams.data.train_slicing,
                                    batch_size=hparams.data.batch_size
                                    )
        val_dataset = CardiacMR2D(hparams.data.val_path,
                                  evaluate=True,
                                  crop_size=hparams.data.crop_size,
                                  slice_range=hparams.data.slice_range,
                                  slicing=hparams.data.val_slicing
                                  )

    else:
        raise ValueError(f'Dataset config ({hparams.data.name}) not recognised.')

    return train_dataset, val_dataset


def worker_init_fn(worker_id):
    """ Callback function passed to DataLoader to initialise the workers """
    # Randomly seed the workers
    random_seed = random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        """Log best metrics whenever a checkpoint is saved"""
        # looks for `hparams` and `hparam_metrics` in `pl_module`
        pl_module.logger.log_metrics(pl_module.hparam_metrics,
                                     step=pl_module.global_step)
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
        }
