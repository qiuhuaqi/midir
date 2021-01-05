import os
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from core_modules.transform.utils import warp, multi_res_warp
from model.utils import get_network, get_transformation, get_loss_fn, get_datasets, worker_init_fn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts

from utils.image import create_img_pyramid
from utils.metric import measure_metrics
from utils.visualise import visualise_result


class LightningDLReg(LightningModule):
    def __init__(self, hparams: DictConfig = None):
        super(LightningDLReg, self).__init__()
        self.hparams = hparams

        self.train_dataset, self.val_dataset = get_datasets(self.hparams)

        self.network = get_network(self.hparams)
        self.transformation = get_transformation(self.hparams)
        self.loss_fn = get_loss_fn(self.hparams)

        # initialise dummy best metrics results for initial logging
        self.hparam_metrics = {f'hparam_metrics_{m}': 0.0
                               for m in self.hparams.meta.hparam_metrics}

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics (for tensorboard HPARAMS)
        self.logger.log_hyperparams(self.hparams, metrics=self.hparam_metrics)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=self.hparams.data.shuffle,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.training.lr_decay_step,
                                                    gamma=0.1,
                                                    last_epoch=-1)
        return [optimizer], [scheduler]

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        out = self.transformation(net_out)
        return out

    def _step(self, batch):
        """ Forward pass inference + compute loss """
        out = self.forward(batch['target'], batch['source'])

        # create image pyramids
        tar_pyr = create_img_pyramid(batch['target'], self.hparams.meta.ml_lvls)
        src_pyr = create_img_pyramid(batch['source'], self.hparams.meta.ml_lvls)

        if self.hparams.transformation.config.svf:
            flows, disps = out
            warped_src_pyr = multi_res_warp(src_pyr, disps)
            losses = self.loss_fn(tar_pyr, warped_src_pyr, flows)

        else:
            disps = out
            warped_src_pyr = multi_res_warp(src_pyr, disps)
            losses = self.loss_fn(tar_pyr, warped_src_pyr, disps)

        step_outputs = {'disp_pred': disps,
                        'target': tar_pyr,
                        'source': src_pyr,
                        'warped_source': warped_src_pyr}
        return losses, step_outputs

    def training_step(self, batch, batch_idx):
        train_losses, _ = self._step(batch)
        self.log_dict({f'train_loss_{k}': loss
                       for k, loss in train_losses.items()})
        return train_losses['loss']

    def validation_step(self, batch, batch_idx):
        # -- Note on data shape -- #
        # 2d:
        #   image data: (N, 1, H, W)
        #   disp gt (synthetic): (N, 2, H, W)
        #  Uses the same way in training and validation: extract one 2D slice from each subject.
        #  This is valid as long as the model doesn't use batch statistics in validation or inference

        # 3d:
        #   image data: (N, 1, H, W, D),
        #   disp gt (synthetic): (N, 3, H, W, D)
        # ------------------------ #

        # run inference, compute losses and outputs
        val_losses, step_outputs = self._step(batch)

        # collect data for measuring metrics
        metric_data = batch
        metric_data.update({k: x[-1] for k, x in step_outputs.items()})
        if 'source_seg' in batch.keys():
            # inference for segmentation metric
            metric_data['warped_source_seg'] = warp(batch['source_seg'], metric_data['disp_pred'],
                                                    interp_mode='nearest')
        if 'target_original' in batch.keys():
            # for visualisation and metrics measuring
            target_original_pyr = create_img_pyramid(batch['target_original'], lvls=self.hparams.meta.ml_lvls)
            step_outputs['target_original'] = target_original_pyr  # for visualisation
            target_pred_pyr = multi_res_warp(target_original_pyr, step_outputs['disp_pred'])
            step_outputs['target_pred'] = target_pred_pyr  # for visualisation
            metric_data['target_pred'] = target_pred_pyr[-1]

        # TODO: metric for 2d cardiac
        # validation metrics
        val_metrics = {k: float(loss.cpu()) for k, loss in val_losses.items()}
        val_metrics.update(measure_metrics(metric_data, self.hparams.meta.metric_groups))

        # log visualisation figure to Tensorboard
        if batch_idx == 0:
            for l in range(self.hparams.meta.ml_lvls):
                # get data for the current resolution level from step_dict
                vis_data_l = dict()
                for k, x in step_outputs.items():
                    vis_data_l[k] = x[l]
                # TODO: visualisation for 2d cardiac
                val_fig_l = visualise_result(vis_data_l, axis=2)
                self.logger.experiment.add_figure(f'val_lvl{l}',
                                                  val_fig_l,
                                                  global_step=self.global_step,
                                                  close=True)
        return val_metrics

    def validation_epoch_end(self, val_metrics):
        """ Process and log accumulated validation results in one epoch """
        val_metrics_epoch = merge_dicts(val_metrics)
        self.log_dict({f'val_metrics_{k}': metric
                       for k, metric in val_metrics_epoch.items()})

        # update hparams metrics
        self.hparam_metrics = {f'hparam_metrics_{k}': val_metrics_epoch[k]
                               for k in self.hparams.meta.hparam_metrics}
