import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from model.lightning import LightningDLReg

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

    # configure logger, checkpoint callback and trainer
    logger = TensorBoardLogger(model_dir, name='log')

    ckpt_callback = ModelCheckpoint(monitor='val_loss',
                                    mode='min',
                                    filepath=f'{logger.log_dir}/checkpoints/'
                                    + '{epoch}-{val_loss:.4f}-{mean_dice_mean:.4f}',
                                    verbose=True
                                    )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      checkpoint_callback=ckpt_callback,
                      **cfg.training.trainer
                      )

    # run training
    trainer.fit(model)


if __name__ == "__main__":
    main()
