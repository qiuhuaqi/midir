import os
import subprocess
from glob import glob

import numpy as np
import torch
from utils.image_io import save_nifti, load_nifti, split_volume_idmat


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, tar, src):
        # identity disp (N, 2, H, W) or (1, 3, H, W, D)
        img_size = tar.size()
        ndim = tar.ndim - 2
        disp = torch.zeros((img_size[0], ndim, *img_size[2:]))
        return disp.type_as(tar)


class MIRTK(object):
    def __init__(
        self,
        mirtk_path=None,
        ds=6,
        model="SVFFD",
        sim="NMI",
        bins=64,
        be=1e-4,
        levels=3,
        work_dir=None,
        verbose=1,
        debug=False,
    ):
        self.model = model
        self.sim = sim
        self.ds = ds
        self.be = be
        self.levels = levels
        self.bins = bins
        self.verbose = verbose

        self.mirtk_path = mirtk_path
        assert mirtk_path is not None, "MIRTK path not specified"
        self.debug = debug

        # configure working directory for MIRTK
        if work_dir is None:
            self.work_dir = os.getcwd() + "/workdir"
        else:
            self.work_dir = work_dir

    def _register(self, tar_path, src_path, dof_path, ndim):
        # Register
        cmd_register = (
            f"{self.mirtk_path} register {tar_path} {src_path} "
            f"-sim {self.sim} "
            f"-model {self.model} "
            f"-ds {self.ds} "
            f"-be {self.be} "
            f"-bins {self.bins} "
            f"-window 7 "
            f"-dofout {dof_path} "
            f"-verbose {self.verbose} "
            f"-levels {self.levels} "
            f"-padding -1 "
        )
        subprocess.check_call(cmd_register, shell=True)

        # Convert dof to disp
        disp_pred_path = self.work_dir + f"/disp_pred.nii.gz"
        cmd_dof_disp = (
            f"{self.mirtk_path} convert-dof {dof_path} {disp_pred_path} "
            f"-output-format disp_voxel "
            f"-target {tar_path}"
        )
        subprocess.check_call(cmd_dof_disp, shell=True)

        # Load converted disp
        if ndim == 2:
            disp = load_nifti(disp_pred_path)[..., 0, 0, :2]  # (H, W, 2)
            disp = np.moveaxis(disp, -1, 0)  # (2, H, W)
        elif ndim == 3:
            disp = load_nifti(disp_pred_path)[..., 0, :]  # (H, W, D, 3)
            disp = np.moveaxis(disp, -1, 0)[np.newaxis, ...]  # (1, 3, H, W, D)
        return disp

    def register2d(self, tar, src):
        """2D registration via splitting the volume to 2D slices,
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
        split_volume_idmat(tar_path, f"{self.work_dir}/tar_z")
        split_volume_idmat(src_path, f"{self.work_dir}/src_z")

        disp_stack = []
        for z in range(len(glob(f"{self.work_dir}/tar_z*"))):
            tar_path_z = f"{self.work_dir}/tar_z{z:02d}.nii.gz"
            src_path_z = f"{self.work_dir}/src_z{z:02d}.nii.gz"
            dof_path_z = f"{self.work_dir}/dof_z{z:02d}.dof.gz"
            disp_z = self._register(tar_path_z, src_path_z, dof_path_z, ndim=2)
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
        disp = self._register(tar_path, src_path, dof_path, ndim=3)
        return disp

    def __call__(self, tar, src):
        ndim = tar.ndim - 2

        # set up work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # perform registration
        if ndim == 2:
            disp = self.register2d(tar, src)
        elif ndim == 3:
            disp = self.register3d(tar, src)
        else:
            raise RuntimeError(f"Unknown dimension {ndim}")

        # clean up work_dir
        if not self.debug:
            os.system(f"rm -r {self.work_dir}")
        return torch.from_numpy(disp).type_as(tar)
