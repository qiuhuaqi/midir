import os
import subprocess
from glob import glob

import numpy as np
import torch
from utils.image_io import save_nifti, load_nifti, split_volume_idmat


class Identity(object):
    def __init__(self, ndim):
        self.dim = ndim

    def __call__(self, tar, src):
        # identity disp (N, 2, H, W) or (1, 3, H, W, D)
        img_size = tar.size()
        disp = torch.zeros((img_size[0], self.dim, *img_size[2:])).type_as(tar)
        return disp


class MIRTK(object):
    """ MIRTK registration by calling $mirtk applications"""
    def __init__(self,
                 mirtk_path=None,
                 ndim=3,
                 ds=6,
                 model='SVFFD',
                 sim='NMI',
                 bins=64,
                 be=1e-4,
                 work_dir=None,
                 debug=False
                 ):
        self.ndim = ndim
        self.model = model
        self.sim = sim
        self.ds = ds
        self.be = be
        self.bins = bins

        self.mirtk_path = mirtk_path
        assert mirtk_path is not None, 'MIRTK path not specified'
        self.debug = debug

        # configure working directory for MIRTK
        if work_dir is None:
            self.work_dir = os.getcwd() + '/workdir'
        else:
            self.work_dir = work_dir

    def _register(self, tar_path, src_path, dof_path):
        # Register
        cmd_register = f'{self.mirtk_path} register {tar_path} {src_path} ' \
                       f'-sim {self.sim} ' \
                       f'-model {self.model} ' \
                       f'-ds {self.ds} ' \
                       f'-be {self.be} ' \
                       f'-bins {self.bins} ' \
                       f'-window 7 ' \
                       f'-dofout {dof_path} ' \
                       f'-levels 3 ' \
                       f'-padding -1 '
        subprocess.check_call(cmd_register, shell=True)

        # Convert dof to disp
        disp_pred_path = self.work_dir + f'/disp_pred.nii.gz'
        cmd_dof_disp = f'{self.mirtk_path} convert-dof {dof_path} {disp_pred_path} ' \
                       f'-output-format disp_voxel ' \
                       f'-target {tar_path}'
        subprocess.check_call(cmd_dof_disp, shell=True)

        # Load converted disp
        if self.ndim == 2:
            disp = load_nifti(disp_pred_path)[..., 0, 0, :2]  # (H, W, 2)
            disp = np.moveaxis(disp, -1, 0)  # (2, H, W)
        elif self.ndim == 3:
            disp = load_nifti(disp_pred_path)[..., 0, :]  # (H, W, D, 3)
            disp = np.moveaxis(disp, -1, 0)[np.newaxis, ...]  # (1, 3, H, W, D)
        return disp

    def register2d(self, tar, src):
        """ 2D registration via splitting the volume to 2D slices,
         input Tensors shape (N, 1, H, W)"""

        # save the target and source image to work_dir
        tar_path = self.work_dir + "/tar.nii.gz"
        src_path = self.work_dir + "/src.nii.gz"
        # (N, 1, H, W) -> (H, W, N)
        tar = tar.cpu().numpy()[:, 0, ...].transpose(1, 2, 0)
        src = src.cpu().numpy()[:, 0, ...].transpose(1, 2, 0)
        save_nifti(tar, tar_path)
        save_nifti(src, src_path)

        # split volume into 2D slices
        split_volume_idmat(tar_path, f'{self.work_dir}/tar_z')
        split_volume_idmat(src_path, f'{self.work_dir}/src_z')

        disp_stack = []
        for z in range(len(glob(f'{self.work_dir}/tar_z*'))):
            tar_path_z = f'{self.work_dir}/tar_z{z:02d}.nii.gz'
            src_path_z = f'{self.work_dir}/src_z{z:02d}.nii.gz'
            dof_path_z = f'{self.work_dir}/dof_z{z:02d}.dof.gz'
            disp_z = self._register(tar_path_z, src_path_z, dof_path_z)
            disp_stack += [disp_z]
        return np.array(disp_stack)

    def register3d(self, tar, src):
        """Execute MIRTK registration of a volume pair,
         input/return Tensors shape (N, 3, H, W, D)"""
        # save the target and source image to work_dir
        tar_path = self.work_dir + "/tar.nii.gz"
        save_nifti(tar.cpu().numpy()[0, 0, ...], tar_path)
        src_path = self.work_dir + "/src.nii.gz"
        save_nifti(src.cpu().numpy()[0, 0, ...], src_path)

        # call MIRTK registration
        dof_path = self.work_dir + "/dof_out.dof.gz"
        disp = self._register(tar_path, src_path, dof_path)
        return disp

    def __call__(self, tar, src):
        assert tar.ndim-2 == self.ndim, \
            f'Data ndim {tar.ndim-2} and module ndim {self.ndim} mismatch.'

        # set up work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # perform registration
        if self.ndim == 2:
            disp = self.register2d(tar, src)
        elif self.ndim == 3:
            disp = self.register3d(tar, src)
        else:
            raise RuntimeError(f'Unknown dimension {self.ndim}')

        # clean up work_dir
        if not self.debug:
            os.system(f'rm -r {self.work_dir}')
        return torch.from_numpy(disp).type_as(tar)
