import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from model.transformation import warp
from model.utils import get_network, get_transformation, get_loss_fn, get_datasets, worker_init_fn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts
from utils.metric import measure_metrics
from utils.visualise import visualise_result


class LightningDLReg(LightningModule):
    def __init__(self, hparams):
        super(LightningDLReg, self).__init__()
        self.hparams = hparams

        self.train_dataset, self.val_dataset = get_datasets(self.hparams)

        self.network = get_network(self.hparams)
        self.transformation = get_transformation(self.hparams)
        self.loss_fn = get_loss_fn(self.hparams)

        # initialise dummy best metrics results for initial logging
        self.hparam_metrics = {f'hparam_metrics/{m}': 0.0
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
        tar = batch['target']
        src = batch['source']
        out = self.forward(tar, src)

        if self.hparams.transformation.config.svf:
            # output the flow field and disp field
            flow, disp = out
            warped_src = warp(src, disp)
            losses = self.loss_fn(tar, warped_src, flow)

        else:
            # only output disp field
            disp = out
            warped_src = warp(src, disp)
            losses = self.loss_fn(tar, warped_src, disp)

        step_outputs = {'disp_pred': disp,
                        'warped_source': warped_src}
        return losses, step_outputs

    def training_step(self, batch, batch_idx):
        train_losses, _ = self._step(batch)
        self.log_dict({f'train_loss/{k}': loss
                       for k, loss in train_losses.items()})
        return train_losses['loss']

    def validation_step(self, batch, batch_idx):
        for k, x in batch.items():
            # reshape data for inference
            # 2d: (N=1, num_slices, H, W) -> (num_slices, N=1, H, W)
            # 3d: (N=1, 1, H, W, D) -> (1, N=1, H, W, D)
            batch[k] = x.transpose(0, 1)

        # run inference, compute losses and outputs
        val_losses, step_outputs = self._step(batch)

        # collect data for measuring metrics and validation visualisation
        val_data = batch
        val_data.update(step_outputs)
        if 'source_seg' in batch.keys():
            val_data['warped_source_seg'] = warp(batch['source_seg'], val_data['disp_pred'],
                                                 interp_mode='nearest')
        if 'target_original' in batch.keys():
            val_data['target_pred'] = warp(val_data['target_original'], val_data['disp_pred'])

        # calculate validation metrics
        val_metrics = {k: float(loss.cpu())
                       for k, loss in val_losses.items()}
        val_metrics.update(measure_metrics(val_data, self.hparams.meta.metric_groups))

        # log visualisation figure to Tensorboard
        if batch_idx == 0:
            val_fig = visualise_result(val_data, axis=2)
            self.logger.experiment.add_figure(f'val_fig',
                                              val_fig,
                                              global_step=self.global_step,
                                              close=True)
        return val_metrics

    def validation_epoch_end(self, val_metrics):
        """ Process and log accumulated validation results in one epoch """
        val_metrics_epoch = merge_dicts(val_metrics)
        self.log_dict({f'val_metrics/{k}': metric
                       for k, metric in val_metrics_epoch.items()})

        # update hparams metrics
        self.hparam_metrics = {f'hparam_metrics/{k}': val_metrics_epoch[k]
                               for k in self.hparams.meta.hparam_metrics}
