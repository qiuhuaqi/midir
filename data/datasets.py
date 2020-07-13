import os
import random

import numpy as np
import torch
import torch.utils.data as ptdata

from data.synthesis import GaussianFilter, synthesis_elastic_deformation
from utils.image import crop_and_pad, normalise_intensity, bbox_from_mask, bbox_crop
from utils.image_io import load_nifti


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


class BrainData(object):
    """
    # todo: this object shouldn't be at this level. Move to config parsing
    Data object:
    - Construct Datasets and Dataloaders
    - Standardize data interface
    """
    def __init__(self, args, params):
        # path check
        train_data_path = params.train_data_path
        assert os.path.exists(train_data_path), f"Training data path does not exist: \n{train_data_path}, not generated?"
        val_data_path = params.val_data_path
        assert os.path.exists(val_data_path), f"Validation data path does not exist: \n{val_data_path}, not generated?"
        test_data_path = params.test_data_path
        assert os.path.exists(test_data_path), f"Testing data path does not exist: \n{test_data_path}, not generated?"

        # training data
        if params.dim == 2:
            # synthesis on-the-fly training data
            self.train_dataset = CamCANSynthDataset(train_data_path,
                                                    dim=params.dim,
                                                    run="train",
                                                    cps=params.synthesis_cps,
                                                    sigma=params.synthesis_sigma,
                                                    disp_max=params.disp_max,
                                                    crop_size=params.crop_size,
                                                    slice_range=tuple(params.slice_range))
            # # load pre-generated training data
            # self.train_dataset = NewBrainDataset(train_data_path, "train", params.dim,
            #                                      slice_range=tuple(params.slice_range))
        elif params.dim == 3:
            # load pre-generated training data
            self.train_dataset = BrainLoadingDataset(train_data_path, "train", params.dim)

        else:
            raise ValueError("Data parsing: dimension of data not specified/recognised.")

        self.train_dataloader = ptdata.DataLoader(self.train_dataset,
                                                  batch_size=params.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.num_workers,
                                                  pin_memory=args.cuda,
                                                  worker_init_fn=worker_init_fn  # todo: fix random seeding
                                                  )


        # val/test data
        self.val_dataset = BrainLoadingDataset(val_data_path, "val", params.dim)
        self.val_dataloader = ptdata.DataLoader(self.val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=args.cuda
                                                )

        self.test_dataset = BrainLoadingDataset(test_data_path, "test", params.dim)
        self.test_dataloader = ptdata.DataLoader(self.test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=args.cuda
                                                 )


class BrainLoadingDataset(ptdata.Dataset):
    """
    Dataset that loads saved generated & pre-processed data
    """
    def __init__(self, data_path, run, dim, slice_range=(70, 90)):
        super(BrainLoadingDataset, self).__init__()
        self.data_path = data_path
        self.run = run  # "train", "val" or "test"
        self.dim = dim
        self.subject_list = sorted(os.listdir(self.data_path))

        self.slice_range = slice_range

    def _load_2d(self, data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            if name == "dvf_gt":
                # dvf is saved in shape (H, W, N, 2) -> (N, 2, H, W)
                data_dict[name] = load_nifti(data_path).transpose(2, 3, 0, 1)

            else:
                # image is saved in shape (H, W, N) ->  (N, H, W)
                data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)

        # randomly select a slice for training
        # todo: this is not used?
        if self.run == "train":
            z = random.randint(self.slice_range[0], self.slice_range[1])
            slicer = slice(z, z+1)  # use slicer to keep dim
            for name, data in data_dict.items():
                data_dict[name] = data[slicer, ...]  # (1, H, W)

        return data_dict

    @staticmethod
    def _load_3d(data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            if name == "dvf_gt":
                # skip loading ground truth DVF for training data

                # dvf_gt is saved in shape (H, W, D, 3) -> (ch=1, 3, H, W, D)
                data_dict[name] = load_nifti(data_path).transpose(3, 0, 1, 2)[np.newaxis, ...]

            else:
                # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
                data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
        return data_dict

    def __getitem__(self, index):

        subject_id = self.subject_list[index]
        data_path_dict = dict()
        for name in ["target", "source", "target_original", "roi_mask", "dvf_gt"]:
             data_path_dict[name] = f"{self.data_path}/{subject_id}/{name}.nii.gz"

        # load 2d/3d
        data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)

        # cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()

        return data_dict

    def __len__(self):
        return len(self.subject_list)


class _SynthDataset(ptdata.Dataset):
    """
    Loading, processing and synthesising transformation
    """
    def __init__(self,
                 run,
                 data_path,
                 dim,
                 sigma=8,
                 cps=10,
                 disp_max=1.,
                 crop_size=192,
                 slice_range=(70, 90),
                 device=torch.device('cpu')
                 ):
        super(_SynthDataset, self).__init__()

        self.data_path = data_path
        self.subject_list = sorted(os.listdir(self.data_path))

        self.run = run
        self.dim = dim
        self.crop_size = crop_size
        self.slice_range = slice_range
        self.device = device

        # elastic parameters
        self.sigma = sigma
        self.cps = cps
        self.disp_max = disp_max

        # Gaussian smoothing filter for random transformation generation
        self.smooth_filter = GaussianFilter(dim=self.dim, sigma=self.sigma)

    @staticmethod
    def _set_path(data_path, subj_id):
        """Sets data keys and paths to data files"""
        raise NotImplementedError

    def _load_2d(self, data_path_dict):
        """2D axial slices, data shape (N=#slices, H, W)"""
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)

        # slice selection
        if self.run == "train":
            # randomly select a slice within range
            z = random.randint(self.slice_range[0], self.slice_range[1])
            slicer = slice(z, z + 1)  # keep dim
        else:  # generate
            # take all slices within range
            slicer = slice(self.slice_range[0], self.slice_range[1])

        for name, data in data_dict.items():
            data_dict[name] = data[slicer, ...]  # (N/1, H, W)

        return data_dict

    @staticmethod
    def _load_3d(data_path_dict):
        """3D volumes, extend shape to (N=1, H, W, D)"""
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
        return data_dict

    @staticmethod
    def _crop_and_pad(data_dict, crop_size):
        # cropping and padding
        for name, data in data_dict.items():
            data_dict[name] = crop_and_pad(data, new_size=crop_size)
        return data_dict

    @staticmethod
    def _normalise_intensity(data_dict):
        # intensity normalisation to [0, 1]
        for name in ["target_original", "source"]:
            data_dict[name] = normalise_intensity(data_dict[name],
                                                  min_out=0., max_out=1.,
                                                  mode="minmax", clip=True)
        return data_dict

    @staticmethod
    def _to_tensor(data_dict):
        # cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()
        return data_dict

    def _synthesis(self, data_dict):
        # generate synthesised DVF and deformed T1 image
        data_dict["target"], data_dict["dvf_gt"] = synthesis_elastic_deformation(data_dict["target_original"],
                                                                                 data_dict["roi_mask"],
                                                                                 smooth_filter=self.smooth_filter,
                                                                                 cps=self.cps,
                                                                                 disp_max=self.disp_max,
                                                                                 device=self.device)
        return data_dict

    def __getitem__(self, index):
        subject_id = self.subject_list[index]
        data_path_dict = self._set_path(self.data_path, subject_id)
        data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)  # load 2d/3d
        data_dict = self._crop_and_pad(data_dict, self.crop_size)
        data_dict = self._normalise_intensity(data_dict)
        data_dict = self._synthesis(data_dict)
        return self._to_tensor(data_dict)

    def __len__(self):
        return len(self.subject_list)


class BratsSynthDataset(_SynthDataset):
    def __init__(self, *args, **kwargs):
        super(BratsSynthDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def _set_path(data_path, subject_id):
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{data_path}/{subject_id}/{subject_id}_t1.nii.gz"
        data_path_dict["source"] = f"{data_path}/{subject_id}/{subject_id}_t2.nii.gz"
        data_path_dict["roi_mask"] = f"{data_path}/{subject_id}/{subject_id}_brainmask.nii.gz"
        return data_path_dict


class IXISynthDataset(_SynthDataset):
    def __init__(self, *args, **kwargs):
        super(IXISynthDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def _set_path(data_path, subj_id):
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{data_path}/{subj_id}/T1-brain.nii.gz"
        data_path_dict["source"] = f"{data_path}/{subj_id}/T2-brain.nii.gz"
        data_path_dict["roi_mask"] = f"{data_path}/{subj_id}/T1-brain_mask.nii.gz"
        return data_path_dict

    @staticmethod
    def _crop_and_pad(data_dict, crop_size):
        # todo: this should be deprecated if IXI data is aligned to MNI space
        # crop by brain mask bounding box for IXI dataset to centre
        bbox, _ = bbox_from_mask(data_dict["roi_mask"], pad_ratio=0.0)
        for name, data in data_dict.items():
            data_dict[name] = bbox_crop(data[:, np.newaxis, ...], bbox)[:, 0, ...]

        # cropping and pad images
        for name in ["target_original", "source", "roi_mask"]:
            data_dict[name] = crop_and_pad(data_dict[name], new_size=crop_size)
        return data_dict


class CamCANSynthDataset(_SynthDataset):
    def __init__(self, *args, **kwargs):
        super(CamCANSynthDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def _set_path(data_path, subj_id):
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{data_path}/{subj_id}/T1_brain.nii.gz"
        data_path_dict["source"] = f"{data_path}/{subj_id}/T2_brain.nii.gz"
        data_path_dict["roi_mask"] = f"{data_path}/{subj_id}/T1_brain_mask.nii.gz"

        # structural segmentation maps
        data_path_dict["cor_seg"] = f"{data_path}/{subj_id}/fsl_cortical_seg.nii.gz"
        data_path_dict["subcor_seg"] = f"{data_path}/{subj_id}/fsl_all_fast_firstseg.nii.gz"
        return data_path_dict