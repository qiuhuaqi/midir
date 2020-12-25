import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from model.lightning import LightningDLReg
from model.utils import MyModelCheckpoint

import random
random.seed(7)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # model_dir set via CLI hydra.run.dir
    model_dir = os.getcwd()

    # use only one GPU
    gpu = cfg.meta.gpu
    if gpu is not None and isinstance(gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # lightning model
    model = LightningDLReg(hparams=cfg)

    # configure logger
    logger = TensorBoardLogger(model_dir, name='log')

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = MyModelCheckpoint(save_last=True,
                                      dirpath=f'{model_dir}/checkpoints/',
                                      # filename='{epoch}_{val_metrics_loss:.4f}'
                                      #          '_{val_metrics_mean_dice:.4f}'
                                      #          '_{val_metrics_folding_ratio:4f}',
                                      verbose=True
                                      )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      callbacks=[ckpt_callback],
                      **cfg.training.trainer
                      )

    # run training
    trainer.fit(model)


if __name__ == "__main__":
    main()
