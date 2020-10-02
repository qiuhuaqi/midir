import os
import subprocess

import numpy as np
import torch
from utils.image_io import save_nifti, load_nifti


class Identity(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tar, src):
        # identity DVF (N, 2, H, W) or (1, 3, H, W, D)
        img_size = tar.size()
        dvf = torch.zeros((img_size[0], self.dim, *img_size[2:])).type_as(tar)
        return dvf


class MirtkFFD(object):
    """" Baseline method: conventional iterative registration using MIRTK"""
    MIRTK_PATH = "/vol/biomedic2/hq615/softwares/mirtk_2.1-pre1/mirtk"

    def __init__(self, hparams):
        self.hparams = hparams

        # configure working directory for MIRTK
        if self.hparams.work_dir is None:
            self.hparams.work_dir = os.getcwd() + '/workdir'
        if not os.path.exists(self.hparams.work_dir):
            os.makedirs(self.hparams.work_dir)

    def __call__(self, tar, src):
        """Execute MIRTK registration of a volume pair, input/return Tensors"""

        # save the target and source image in work dir
        tar_path = self.hparams.work_dir + "/target.nii.gz"
        save_nifti(tar.cpu().numpy()[0, 0, ...], tar_path)

        src_path = self.hparams.work_dir + "/source.nii.gz"
        save_nifti(src.cpu().numpy()[0, 0, ...], src_path)

        # dof output path
        dof_path = self.hparams.work_dir + '/dof.dof.gz'

        # register
        cmd_register = f'{MirtkFFD.MIRTK_PATH} register {tar_path} {src_path} ' \
                       f'-sim {self.hparams.sim} ' \
                       f'-model {self.hparams.model} ' \
                       f'-ds {self.hparams.ds} ' \
                       f'-be {self.hparams.be} ' \
                       f'-bins {self.hparams.bins} ' \
                       f'-window 7 ' \
                       f'-dofout {dof_path} ' \
                       f'-model FFD ' \
                       f'-levels 3 ' \
                       f'-padding -1 '
        subprocess.check_call(cmd_register, shell=True)

        # convert dof to dvf
        dvf_pred_path = self.hparams.work_dir + f'/dvf_pred.nii.gz'
        cmd_dof_dvf = f'{MirtkFFD.MIRTK_PATH} convert-dof {dof_path} {dvf_pred_path} ' \
                      f'-output-format disp_voxel ' \
                      f'-target {tar_path}'
        subprocess.check_call(cmd_dof_dvf, shell=True)

        # load converted dvf
        dvf = load_nifti(dvf_pred_path)[..., 0, :]  # (H, W, D, 3)
        dvf = np.moveaxis(dvf, -1, 0)[np.newaxis, ...]  # (1, 3, H, W, D)

        return torch.from_numpy(dvf).type_as(tar)



class AntsSyN(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def __call__(self, tar, src):
        # dvf = ants_syn_reg(tar, src)
        dvf = None
        # dvf = dvf.type_as(tar)
        return dvf
