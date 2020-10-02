import os
import random
import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# from data.datasets import CamCANSynthDataset, BrainLoadingDataset
from data.datasets import BrainInterSubject3DTrain, BrainInterSubject3DEval

from model.transformations import spatial_transform
from model.utils import get_network, get_transformation, get_loss_fn

from utils.image import bbox_from_mask
from utils.metric import measure_metrics
from utils.visualise import visualise_result

import pytorch_lightning as pl


def worker_init_fn(worker_id):
    """Callback function passed to DataLoader to initialise the workers"""
    # # generate a random sequence of seeds for the workers
    # print(f"Random state before generating the random seed: {random.getstate()}")
    random_seed = random.randint(0, 2 ** 32 - 1)
    # ##debug
    # print(f"Random state after generating the random seed: {random.getstate()}")
    # print(f"Random seed for worker {worker_id} is: {random_seed}")
    # ##
    np.random.seed(random_seed)


class LightningDLReg(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None):
        super(LightningDLReg, self).__init__()
        self.hparams = hparams

        self.network = get_network(self.hparams)
        self.transformation = get_transformation(self.hparams)
        self.sim_loss_fn, self.reg_loss_fn = get_loss_fn(self.hparams)

        # initialise best metric
        self.best_metric_result = self.hparams.meta.best_metric_init

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics
        best_metric_init = {'best/' + self.hparams.meta.best_metric: self.hparams.meta.best_metric_init}
        self.logger.log_hyperparams(self.hparams, metrics=best_metric_init)

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
                          worker_init_fn=worker_init_fn  # todo: fix random seeding
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
                          pin_memory=self.on_gpu
                          )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.training.lr)

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        dvf = self.transformation(net_out)
        return dvf

    def _roi_crop(self, x, mask):
        bbox, _ = bbox_from_mask(mask.squeeze(1).cpu().numpy())
        # TODO: 1) add Tensor support, 2) move this to utility
        for i in range(self.hparams.data.dim):
            x = x.narrow(i + 2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))
        # returns a view instead of a copy?
        return x

    def loss_fn(self, batch, warped_source, dvf_pred):
        target = batch['target']

        # (optional) only evaluate similarity loss within ROI bounding box
        if self.hparams.loss.loss_roi:
            assert batch['roi_mask'] is not None, "Loss ROI mask not provided."
            mask = batch['roi_mask']
            target = self._roi_crop(target, mask)
            warped_source = self._roi_crop(warped_source, mask)

        # compute similarity and regularisation losses
        sim_loss = self.sim_loss_fn(target, warped_source) * self.hparams.loss.sim_weight
        reg_loss = self.reg_loss_fn(dvf_pred) * self.hparams.loss.reg_weight

        losses = {"loss": sim_loss + reg_loss,
                  self.hparams.loss.sim_loss: sim_loss,
                  self.hparams.loss.reg_loss: reg_loss}
        return losses

    def _step(self, batch):
        """ Forward pass inference + compute loss """
        dvf_pred = self.forward(batch['target'], batch['source'])
        warped_source = spatial_transform(batch['source'], dvf_pred)
        losses = self.loss_fn(batch, warped_source, dvf_pred)

        step_dict = {'dvf_pred': dvf_pred,
                     'warped_source': warped_source}
        return losses, step_dict

    def training_step(self, batch, batch_idx):
        train_losses, step_dict = self._step(batch)

        # training logs
        if self.global_step % self.trainer.row_log_interval == 0:
            for k, loss in train_losses.items():
                # self.logger.experiment.add_scalars(k, {'train': loss}, global_step=self.global_step)
                self.logger.experiment.add_scalar(f'train_loss/{k}',
                                                  loss,
                                                  global_step=self.global_step)
        return {'loss': train_losses['loss']}

    def validation_step(self, batch, batch_idx):
        # reshape data from dataloader
        # TODO: hacky reshape data is not compatible with 3D when batch_size>1
        for k, x in batch.items():
            if k == "dvf_gt":
                batch[k] = x[0, ...]  # (N, dim, *(sizes))
            else:
                batch[k] = x.transpose(0, 1)  # (N, 1, *(dims))

        # run inference, compute losses and outputs
        val_losses, step_dict = self._step(batch)

        if 'source_seg' in batch.keys():
            # apply estimated transformation to segmentation
            step_dict['warped_source_seg'] = spatial_transform(batch['source_seg'], step_dict['dvf_pred'],
                                                               interp_mode='nearest')

        if 'target_original' in batch.keys():
            step_dict['target_pred'] = spatial_transform(batch['target_original'], step_dict['dvf_pred'])

        # calculate metric for one validation batch
        val_data = dict(batch, **step_dict)  # merge data dicts
        metric_result_step = measure_metrics(val_data,
                                             self.hparams.meta.metric_groups,
                                             return_tensor=True)

        # log one figure to visually monitor training
        if batch_idx == 0:
            val_fig = visualise_result(val_data, axis=2)
            self.logger.experiment.add_figure('val',
                                              val_fig,
                                              global_step=self.global_step,
                                              close=True)
        return val_losses, metric_result_step

    def _log_metrics(self, metric_result):
        """ Log metrics """
        if self.hparams.meta.metrics_to_log is not None:
            # log only selected metrics
            for metric in self.hparams.meta.metrics_to_log:
                self.logger.experiment.add_scalar(f'metrics/{metric}_mean',
                                                  metric_result[metric + '_mean'],
                                                  global_step=self.global_step)
                self.logger.experiment.add_scalar(f'metrics/{metric}_std',
                                                  metric_result[metric + '_std'],
                                                  global_step=self.global_step)
        else:
            for k, x in metric_result.items():
                self.logger.experiment.add_scalar(f'metrics/{k}',
                                                  x,
                                                  global_step=self.global_step)

    def _check_log_best_metric(self, metric_result):
        """ Update the best metric """
        current = metric_result[self.hparams.meta.best_metric]

        if self.hparams.meta.best_metric_mode == 'max':
            is_best = current > self.best_metric_result
        elif self.hparams.meta.best_metric_mode == 'min':
            is_best = current < self.best_metric_result
        else:
            is_best = False

        if is_best:
            # update best metric value
            self.logger.experiment.add_scalar('best/' + self.hparams.meta.best_metric,
                                              current,
                                              global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        # reduce and log validation loss
        losses_list = [x[0] for x in outputs]
        losses_reduced = dict()
        for k in losses_list[0].keys():
            loss_reduced = torch.stack([x[k] for x in losses_list]).mean()
            losses_reduced[k] = loss_reduced
            self.logger.experiment.add_scalar(f'val_loss/{k}',
                                              loss_reduced,
                                              global_step=self.global_step)

        # reduce and log validation metric results (mean & std)
        metric_result_list = [x[1] for x in outputs]
        metric_result_reduced = dict()
        for k in metric_result_list[0].keys():
            stacked = torch.stack([x[k] for x in metric_result_list])
            metric_result_reduced[f'{k}_mean'] = stacked.mean()
            metric_result_reduced[f'{k}_std'] = stacked.std()

        # log metric results and update best metrics
        self._log_metrics(metric_result_reduced)
        self._check_log_best_metric(metric_result_reduced)

        # return callback metrics for checkpointing
        return {'val_loss': losses_reduced['loss'], 'mean_dice_mean': metric_result_reduced['mean_dice_mean']}
