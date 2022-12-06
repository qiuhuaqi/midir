import os
import random
from typing import Dict, Any

import numpy as np
import omegaconf
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint


def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param


def save_dict_to_csv(d, csv_path, model_name="modelX"):
    for k, x in d.items():
        if not isinstance(x, list):
            d[k] = [x]
    pd.DataFrame(d, index=[model_name]).to_csv(csv_path)


def worker_init_fn(worker_id):
    """Callback function passed to DataLoader to initialise the workers"""
    # Randomly seed the workers
    random_seed = random.randint(0, 2**32 - 1)
    np.random.seed(random_seed)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        """Log best metrics whenever a checkpoint is saved"""
        # looks for `hparams` and `hparam_metrics` in `pl_module`
        pl_module.logger.log_metrics(
            pl_module.hparam_metrics, step=pl_module.global_step
        )
        self.state_dict().update(
            {
                "monitor": self.monitor,
                "best_model_score": self.best_model_score,
                "best_model_path": self.best_model_path,
                "current_score": self.current_score,
            }
        )
