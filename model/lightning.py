import os
import random
import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from data.datasets import BrainInterSubject3DTrain, BrainInterSubject3DEval

from core_modules.transform.utils import warp, multi_res_warp
from model.utils import get_network, get_transformation, get_loss_fn

from utils.image import create_img_pyramid
from utils.metric import measure_metrics
from utils.visualise import visualise_result

import pytorch_lightning as pl


def worker_init_fn(worker_id):
    """ Callback function passed to DataLoader to initialise the workers """
    # Randomly seed the workers
    random_seed = random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)


class LightningDLReg(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None):
        super(LightningDLReg, self).__init__()
        self.hparams = hparams

        self.network = get_network(self.hparams)
        self.transformation = get_transformation(self.hparams)
        self.loss_fn = get_loss_fn(self.hparams)

        # initialise dummy best metrics results for initial logging
        self.best_metric_result = {f'best/{m}': 0.0 for m in self.hparams.meta.best_metrics}

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics (for tensorboard HPARAMS)
        self.logger.log_hyperparams(self.hparams, metrics=self.best_metric_result)

    def train_dataloader(self):
        assert os.path.exists(self.hparams.data.train_path), \
            f"Training data path does not exist: {self.hparams.data.train_path}"

        train_dataset = BrainInterSubject3DTrain(self.hparams.data.train_path,
                                                 self.hparams.data.crop_size,
                                                 modality=self.hparams.data.modality,
                                                 atlas_path=self.hparams.data.atlas_path)

        return DataLoader(train_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=self.hparams.data.shuffle,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn
                          )

    def val_dataloader(self):
        assert os.path.exists(self.hparams.data.val_path), \
            f"Validation data path does not exist: {self.hparams.data.val_path}"

        val_dataset = BrainInterSubject3DEval(self.hparams.data.val_path,
                                              self.hparams.data.crop_size,
                                              modality=self.hparams.data.modality,
                                              atlas_path=self.hparams.data.atlas_path)

        return DataLoader(val_dataset,
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

        step_dict = {'disp_pred': disps,
                     'target': tar_pyr,
                     'source': src_pyr,
                     'warped_source': warped_src_pyr}
        return losses, step_dict

    def training_step(self, batch, batch_idx):
        train_losses, _ = self._step(batch)

        # training logs
        if self.global_step % self.trainer.row_log_interval == 0:
            for k, loss in train_losses.items():
                # self.logger.experiment.add_scalars(k, {'train': loss}, global_step=self.global_step)
                self.logger.experiment.add_scalar(f'train_loss/{k}',
                                                  loss.detach(),
                                                  global_step=self.global_step)
        return {'loss': train_losses['loss']}

    def validation_step(self, batch, batch_idx):
        # reshape data from dataloader
        # TODO: hacky reshape data is not compatible with 3D when batch_size>1
        for k, x in batch.items():
            if k == "disp_gt":
                batch[k] = x[0, ...]  # (N, dim, *(sizes))
            else:
                batch[k] = x.transpose(0, 1)  # (N, 1, *(dims))

        # run inference, compute losses and outputs
        val_losses, step_dict = self._step(batch)

        # metrics are evaluated using data of the original resolution
        metric_data = {k: x[-1] for k, x in step_dict.items()}

        if 'source_seg' in batch.keys():
            # inference for segmentation metric
            metric_data['warped_source_seg'] = warp(batch['source_seg'], metric_data['disp_pred'],
                                                    interp_mode='nearest')
        if 'target_original' in batch.keys():
            # for visualisation and metrics measuring
            target_original_pyr = create_img_pyramid(batch['target_original'], lvls=self.hparams.meta.ml_lvls)
            step_dict['target_original'] = target_original_pyr  # for visualisation
            target_pred_pyr = multi_res_warp(target_original_pyr, step_dict['disp_pred'])
            step_dict['target_pred'] = target_pred_pyr  # for visualisation
            metric_data['target_pred'] = target_pred_pyr[-1]

        # measure metrics
        metric_data.update(batch)  # add input data to metric data
        metric_result_step = measure_metrics(metric_data,
                                             self.hparams.meta.metric_groups,
                                             return_tensor=True)

        # log visualisation figure to Tensorboard
        if batch_idx == 0:
            for l in range(self.hparams.meta.ml_lvls):
                # get data for the current resolution level from step_dict
                vis_data_l = dict()
                for k, x in step_dict.items():
                    vis_data_l[k] = x[l]
                val_fig_l = visualise_result(vis_data_l, axis=2)
                self.logger.experiment.add_figure(f'val_lvl{l}',
                                                  val_fig_l,
                                                  global_step=self.global_step,
                                                  close=True)
        return val_losses, metric_result_step

    def _check_log_best_metric(self, val_loss, metric_result):
        """ Update the best metric results, called at validation_epoch_end """
        # TODO: this should be in model checkpointing callback

        current_metric_result = dict()
        current_metric_result.update(val_loss)
        current_metric_result.update(metric_result)

        best_metric = 'loss'  # log best metrics when validation loss is a new lowest
        if self.current_epoch == 0:
            # log the first metric as the best
            is_best = True
        else:
            # determine if the current metric result is the best
            is_best = current_metric_result[best_metric] < self.best_metric_result[best_metric]

        if is_best:
            # update the best metric value and log the other best metrics at this step
            for m in self.hparams.meta.best_metrics:
                self.best_metric_result[m] = current_metric_result[m]
                self.logger.experiment.add_scalar(f'best/{m}',
                                                  current_metric_result[m],
                                                  global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        """ Process and log accumulated validation results in one epoch """
        # average validation loss and log
        val_loss_list = [x[0] for x in outputs]
        val_loss_reduced = dict()
        for k in val_loss_list[0].keys():
            val_loss_reduced[k] = torch.stack([x[k].detach() for x in val_loss_list]).mean()
            self.logger.experiment.add_scalar(f'val_loss/{k}',
                                              val_loss_reduced[k],
                                              global_step=self.global_step)

        # reduce and log validation metric results (mean & std)
        metric_result_list = [x[1] for x in outputs]
        metric_result_reduced = dict()
        for k in metric_result_list[0].keys():
            stacked = torch.stack([x[k] for x in metric_result_list])
            metric_result_reduced[f'{k}_mean'] = stacked.mean()
            metric_result_reduced[f'{k}_std'] = stacked.std()
            self.logger.experiment.add_scalar(f'metrics/{k}_mean',
                                              metric_result_reduced[f'{k}_mean'],
                                              global_step=self.global_step)
            self.logger.experiment.add_scalar(f'metrics/{k}_std',
                                              metric_result_reduced[f'{k}_std'],
                                              global_step=self.global_step)

        # log and update best metrics
        self._check_log_best_metric(val_loss_reduced, metric_result_reduced)

        # return callback metrics for checkpointing
        return {'val_loss': val_loss_reduced['loss'],
                'mean_dice_mean': metric_result_reduced['mean_dice_mean']}
