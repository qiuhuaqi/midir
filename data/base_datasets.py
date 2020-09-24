import os
import random

import numpy as np
import torch
from torch.utils import data as ptdata

from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti


class _BaseDataset(ptdata.Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))

    @staticmethod
    def _to_tensor(data_dict):
        # cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()
        return data_dict

    @staticmethod
    def _crop_and_pad(data_dict, crop_size):
        # cropping and padding
        for name, data in data_dict.items():
            data_dict[name] = crop_and_pad(data, new_size=crop_size)
        return data_dict

    @staticmethod
    def _normalise_intensity(data_dict, keys, vmin=0., vmax=1.):
        for name in keys:
            data_dict[name] = normalise_intensity(data_dict[name],
                                                  min_out=vmin, max_out=vmax,
                                                  mode="minmax", clip=True)
        return data_dict

    def __len__(self):
        return len(self.subject_list)


class _BaseDataset2D(_BaseDataset):
    def __init__(self, data_dir_path, slice_range=(70, 90)):
        super(_BaseDataset2D, self).__init__(data_dir_path=data_dir_path)
        self.slice_range = slice_range

    def _load(self, data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            # image is saved in shape (H, W, N) ->  (N, H, W)
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)

        # randomly select a slice for training
        z = random.randint(self.slice_range[0], self.slice_range[1])
        slicer = slice(z, z + 1)  # use slicer to keep dim
        for name, data in data_dict.items():
            data_dict[name] = data[slicer, ...]  # (1, H, W)
        return data_dict

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError


class _BaseDataset3D(_BaseDataset):
    def __init__(self, data_dir_path):
        super(_BaseDataset3D, self).__init__(data_dir_path=data_dir_path)

    @staticmethod
    def _load(data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
            data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
        return data_dict

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError
