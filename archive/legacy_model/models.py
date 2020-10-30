import subprocess

import torch
import torch.nn as nn
import numpy as np

from model.network.networks import UNet, FFDNet, SiameseNetFFD, SiameseNet
from model.transformations import BSplineFFDTransform, DVFTransform
from utils.image_io import save_nifti, load_nifti


class DLRegModel(nn.Module):
    def __init__(self, params):
        super(DLRegModel, self).__init__()

        self.params = params

        # initialise recording variables
        self.epoch_num = 0
        self.iter_num = 0
        self.is_best = False
        self.best_metric_result = 0

        self._set_network()
        self._set_transformation_model()

    def _set_network(self):
        if self.params.network == "SiameseNetDVF":
            self.network = SiameseNet()

        elif self.params.network == "UNetDVF":
            self.network = UNet(dim=self.params.ndim,
                                enc_channels=self.params.enc_channels,
                                dec_channels=self.params.dec_channels,
                                out_channels=self.params.out_channels
                                )

        elif self.params.network == "SiameseNetFFD":
            self.network = SiameseNetFFD()

        elif self.params.network == "FFDNet":
            self.network = FFDNet(dim=self.params.ndim,
                                  img_size=self.params.crop_size,
                                  cpt_spacing=self.params.ffd_sigma,
                                  enc_channels=self.params.enc_channels,
                                  out_channels=self.params.out_channels
                                  )
        else:
            raise ValueError("Model: Network not recognised")

    def _set_transformation_model(self):
        if self.params.transformation == "DVF":
            self.transform = DVFTransform()

        elif self.params.transformation == "FFD":
            self.transform = BSplineFFDTransform(dim=self.params.ndim,
                                                 img_size=self.params.crop_size,
                                                 cps=self.params.ffd_sigma)
        else:
            raise ValueError("Model: Transformation model not recognised")

    def update_best_model(self, metric_results):
        metric_results_mean = np.mean([metric_results[metric] for metric in self.params.best_metrics])

        if self.epoch_num + 1 == self.params.val_epochs:
            # initialise for the first validation
            self.best_metric_result = metric_results_mean
        else:
            if metric_results_mean < self.best_metric_result:
                self.is_best = True
                self.best_metric_result = metric_results_mean

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        dvf = self.transform(net_out)
        return dvf


class _BaselineModel(nn.Module):
    # todo: this should be the base class for all modules, including the DL ones
    #   adapt this later with Pytorch Lightning
    def __init__(self, params):
        super(_BaselineModel, self).__init__()
        self.params = params

    def forward(self, tar, src):
        raise NotImplementedError


class IdBaselineModel(_BaselineModel):
    """Identity transformation baseline, i.e. no registration"""
    def __init__(self, params):
        super(IdBaselineModel, self).__init__(params)

    def forward(self, tar, src):
        """Output dvf.yaml in shape (N, dim, *(dims))"""
        dim = len(tar.size()) - 2   # image dimension
        dvf = torch.zeros_like(tar).repeat(1, dim, *(1,) * dim)
        return dvf


class MIRTKBaselineModel(_BaselineModel):
    """" Baseline method: conventional iterative registration using MIRTK"""
    def __init__(self, params, work_dir):
        super(MIRTKBaselineModel, self).__init__(params)
        self.work_dir = work_dir

    def forward(self, tar, src):
        """Execute MIRTK registration of a volume pair"""

        # save the target and source image in work dir
        tar_path = self.work_dir + "/target.nii.gz"
        save_nifti(tar.cpu().numpy()[0, 0, ...], tar_path)

        src_path = self.work_dir + "/source.nii.gz"
        save_nifti(src.cpu().numpy()[0, 0, ...], src_path)

        # dof output path
        dof_path = self.work_dir + '/dof.dof.gz'

        # register
        cmd_register = f'mirtk register {tar_path} {src_path} ' \
                       f'-sim {self.params.sim} ' \
                       f'-ds {self.params.ds} ' \
                       f'-be {self.params.be} ' \
                       f'-bins {self.params.bins} ' \
                       f'-dofout {dof_path} ' \
                       f'-model FFD ' \
                       f'-levels 3 ' \
                       f'-padding -1 '
        subprocess.check_call(cmd_register, shell=True)

        # convert dof to dvf.yaml
        dvf_pred_path = self.work_dir + f'/dvf_pred.nii.gz'
        cmd_dof_dvf = f'mirtk convert-dof {dof_path} {dvf_pred_path} ' \
                      f'-output-format disp_voxel ' \
                      f'-target {tar_path}'
        subprocess.check_call(cmd_dof_dvf, shell=True)

        # load converted dvf.yaml
        # todo: only work for 3D a.t.m.
        dvf = load_nifti(dvf_pred_path)[..., 0, :]  # (H, W, D, 3)
        dvf = np.moveaxis(dvf, -1, 0)[np.newaxis, ...]  # (1, 3, H, W, D)
        return torch.from_numpy(dvf).to(device=tar.device)
