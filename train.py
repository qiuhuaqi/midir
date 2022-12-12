import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model.lightning import LightningDLReg

import random

random.seed(7)


@hydra.main(config_path="conf/train", config_name="config")
def main(cfg: DictConfig) -> None:

    # set via CLI hydra.run.dir
    model_dir = os.getcwd()

    # use only one GPU
    gpus = None if cfg.gpu is None else 1
    if isinstance(cfg.gpu, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    # lightning model
    model = LightningDLReg(**cfg)

    # configure logger
    logger = TensorBoardLogger(model_dir, name="log", default_hp_metric=False)

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = ModelCheckpoint(
        dirpath=f"{model_dir}/checkpoints/",
        filename="epoch_{epoch}-dice_{val_metrics/mean_dice:.3f}_folding_{val_metrics/folding_ratio:.3f}",
        save_last=True,
        monitor="val_metrics/mean_dice",
        mode="max",
        save_top_k=2,
        auto_insert_metric_name=False,
        verbose=True,
    )

    trainer = Trainer(
        default_root_dir=model_dir,
        logger=logger,
        callbacks=[ckpt_callback],
        accelerator="gpu",
        devices=1,
        **cfg.training.trainer,
    )

    trainer.fit(model)
    # run validation after the last epoch which should reflect the last checkpoint
    trainer.validate(model)


if __name__ == "__main__":
    main()
