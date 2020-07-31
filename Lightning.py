import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from data.datasets import CamCANSynthDataset, BrainLoadingDataset
from model.networks.dvf_nets import SiameseNet, UNet
from model.networks.ffd_nets import FFDNet
from model.transformations import DVFTransform, BSplineFFDTransform, spatial_transform
from model.losses import MILossGaussian, diffusion_loss, bending_energy_loss
from utils.misc import worker_init_fn, Params
from utils.image import bbox_from_mask
from utils.metric import calculate_metrics
from utils.visualise import visualise_result

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_network(params):
    if params.network == "SiameseNetDVF":
        network = SiameseNet()

    elif params.network == "UNetDVF":
        network = UNet(dim=params.dim,
                       enc_channels=params.enc_channels,
                       dec_channels=params.dec_channels,
                       out_channels=params.out_channels
                       )

    elif params.network == "FFDNet":
        network = FFDNet(dim=params.dim,
                         img_size=params.crop_size,
                         cpt_spacing=params.ffd_sigma,
                         enc_channels=params.enc_channels,
                         out_channels=params.out_channels
                         )
    else:
        raise ValueError("Model building: Network not recognised")
    return network


def get_transformation(params):
    if params.transformation == "DVF":
        transformation = DVFTransform()

    elif params.transformation == "FFD":
        transformation = BSplineFFDTransform(dim=params.dim,
                                             img_size=params.crop_size,
                                             sigma=params.ffd_sigma)
    else:
        raise ValueError("Model: Transformation model not recognised")
    return transformation


def get_loss_fn(params):
    # similarity loss
    if params.sim_loss == 'MSE':
        sim_loss_fn = nn.MSELoss()
    elif params.sim_loss == 'NMI':
        sim_loss_fn = MILossGaussian(num_bins_tar=params.mi_num_bins,
                                     num_bins_src=params.mi_num_bins,
                                     sigma_tar=params.mi_sigma,
                                     sigma_src=params.mi_sigma)
    else:
        raise ValueError(f'Similarity loss not recognised: {params.sim_loss}.')

    # regularisation loss
    if params.reg_loss == "diffusion":
        reg_loss_fn = diffusion_loss
    elif params.reg_loss == "be":
        reg_loss_fn = bending_energy_loss
    else:
        raise ValueError(f"Regularisation loss not recognised: {params.reg_loss}.")

    return sim_loss_fn, reg_loss_fn


class LightningDLReg(pl.LightningModule):
    def __init__(self, args, params):
        super(LightningDLReg, self).__init__()
        self.args = args
        self.params = params

        self.network = get_network(self.params)
        self.transformation = get_transformation(self.params)

    def train_dataloader(self):
        assert os.path.exists(self.params.train_data_path), \
            f"Training data path does not exist: \n{self.params.train_data_path}, not generated?"

        # training data
        if self.params.dim == 2:
            # synthesis on-the-fly training data
            train_dataset = CamCANSynthDataset(data_path=self.params.train_data_path,
                                               dim=self.params.dim,
                                               run="train",
                                               cps=self.params.synthesis_cps,
                                               sigma=self.params.synthesis_sigma,
                                               disp_max=self.params.disp_max,
                                               crop_size=self.params.crop_size,
                                               slice_range=tuple(self.params.slice_range))

        elif self.params.dim == 3:
            # load pre-generated training data
            train_dataset = BrainLoadingDataset(data_path=self.params.train_data_path,
                                                run="train",
                                                dim=self.params.dim,
                                                data_pair=self.params.data_pair,
                                                atlas_path=self.params.atlas_path)

        else:
            raise ValueError("Data parsing: dimension of data not specified/recognised.")

        return DataLoader(train_dataset,
                          batch_size=self.params.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn  # todo: fix random seeding
                          )

    def val_dataloader(self):
        assert os.path.exists(self.params.val_data_path), \
            f"Validation data path does not exist: \n{self.params.val_data_path}, not generated?"
        val_dataset = BrainLoadingDataset(data_path=self.params.val_data_path,
                                          run="val",
                                          dim=self.params.dim,
                                          data_pair=self.params.data_pair,
                                          atlas_path=self.params.atlas_path)
        return DataLoader(val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=self.on_gpu
                          )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.params.learning_rate)

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        dvf = self.transformation(net_out)
        return dvf

    def loss_fn(self, batch, warped_source, dvf_pred):
        # configure loss functions
        sim_loss_fn, reg_loss_fn = get_loss_fn(self.params)

        # (optional) only evaluate similarity loss within ROI bounding box
        # TODO: modularise this
        if self.params.loss_roi:
            assert batch['roi_mask'] is not None, "Loss ROI mask not provided."
            # TODO: add Tensor support
            bbox, _ = bbox_from_mask(batch['roi_mask'].squeeze(1).cpu().numpy())
            for i in range(self.params.dim):
                batch['target'] = batch['target'].narrow(i + 2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))
                warped_source = warped_source.narrow(i + 2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))

        # compute similarity and regularisation losses
        sim_loss = sim_loss_fn(batch['target'], warped_source) * self.params.sim_weight
        reg_loss = reg_loss_fn(dvf_pred) * self.params.reg_weight
        return {"loss": sim_loss + reg_loss,
                self.params.sim_loss: sim_loss,
                self.params.reg_loss: reg_loss}

    def _step(self, batch):
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
                self.logger.experiment.add_scalars(k, {'train': loss}, global_step=self.global_step)
                # self.logger.experiment.add_scalar('train_loss/{k}', loss, global_step=self.global_step)

        return {'loss': train_losses['loss']}

    def validation_step(self, batch, batch_idx):
        # reshape data (to be compatible with 2D)
        for k, x in batch.items():
            if k == "dvf_gt":
                batch[k] = x[0, ...]  # (N, dim, *(dims))
            else:
                batch[k] = x.transpose(0, 1)  # (N, 1, *(dims))

        # inference
        val_losses, step_dict = self._step(batch)
        step_dict["target_pred"] = spatial_transform(batch["target_original"], step_dict["dvf_pred"])

        # calculate metric for one validation batch
        val_data = dict(batch, **step_dict)  # merge data dicts
        metric_result_step = calculate_metrics(val_data, self.params.metric_groups, return_tensor=True)

        # log one validation visual result
        if batch_idx == 0:
            val_fig = visualise_result(val_data, axis=2)
            self.logger.experiment.add_figure('val', val_fig, global_step=self.global_step, close=True)

        return val_losses, metric_result_step

    def validation_epoch_end(self, outputs):
        # reduce and log validation loss
        losses_list = [x[0] for x in outputs]
        losses_reduced = dict()
        for k in losses_list[0].keys():
            loss_reduced = torch.stack([x[k] for x in losses_list]).mean()
            losses_reduced[k] = loss_reduced
            self.logger.experiment.add_scalars(k, {'val': loss_reduced}, global_step=self.global_step)

        # reduce and log validation metric results (mean & std)
        metric_result_list = [x[1] for x in outputs]
        metric_result_reduced = dict()
        for k in metric_result_list[0].keys():
            stacked = torch.stack([x[k] for x in metric_result_list])
            metric_result_reduced[f'{k}_mean'] = stacked.mean()
            metric_result_reduced[f'{k}_std'] = stacked.std()

        # log metric results
        self.logger.log_metrics(metric_result_reduced, step=self.global_step)

        # return callback metrics for checkpointing
        return {'val_loss': losses_reduced['loss'], 'dice_mean': metric_result_reduced['dice_mean']}


def main(args, params):
    # Lightning model and training
    model = LightningDLReg(args, params)
    logger = TensorBoardLogger(args.model_dir, name='log')

    ckpt_callback = ModelCheckpoint(monitor='dice_mean',
                                    mode='max',
                                    filepath=f'{logger.log_dir}/checkpoints/'+'{epoch}-{val_loss:.2f}-{dice_mean:.2f}',
                                    verbose=True)

    # TODO: group trainer arguments
    from pytorch_lightning import Trainer
    trainer = Trainer(default_root_dir=args.model_dir,
                      gpus=1,
                      max_epochs=10,
                      logger=logger,
                      row_log_interval=1,
                      limit_train_batches=3,
                      limit_val_batches=3,
                      check_val_every_n_epoch=1,
                      fast_dev_run=args.debug,
                      checkpoint_callback=ckpt_callback)
    trainer.fit(model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r',
                        default='train',
                        help="'train' or 'eval'")


    parser.add_argument('--model_dir', '-m',
                        default='experiments/base_model',
                        help="Directory containing params.json")

    parser.add_argument('--gpu', '-g',
                        default=0, type=int,
                        help='Choose GPU')

    parser.add_argument('--save',
                        action='store_true',
                        help="Save deformed images and predicted DVFs in evaluation if True")

    parser.add_argument('--ckpt_file',
                        default=None,
                        help="Name of the checkpoint file in model_dir:"
                             " 'best.pth.tar' for best model, or 'last.pth.tar' for the last saved checkpoint")

    parser.add_argument('--cpu',
                        action='store_true',
                        help='Use CPU')

    parser.add_argument('--num_workers',
                        default=2,

                        type=int,
                        help='Number of processes used by dataloader, 0 means use main process')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.gpu is not None and isinstance(args.gpu, int):
        # only using 1 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # load config parameters from the JSON file
    # todo: use Hydra instead
    json_path = args.model_dir + '/params.json'
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = Params(json_path)
