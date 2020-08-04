import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from Lightning import LightningDLReg
from pytorch_lightning import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    # model_dir via CLI: hydra.run.dir
    model_dir = os.getcwd()

    # use only one GPU
    gpu = cfg.meta.gpu
    if gpu is not None and isinstance(gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    model = LightningDLReg(hparams=cfg)

    logger = TensorBoardLogger(model_dir, name='log')
    ckpt_callback = ModelCheckpoint(monitor=cfg.meta.best_metric,
                                    mode=cfg.meta.best_metric_mode,
                                    filepath=f'{logger.log_dir}/checkpoints/'+'{epoch}-{val_loss:.2f}-{dice_mean:.2f}',
                                    verbose=True
                                    )
    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      checkpoint_callback=ckpt_callback,
                      **cfg.training.trainer
                      )
    trainer.fit(model)


if __name__ == "__main__":
    main()
