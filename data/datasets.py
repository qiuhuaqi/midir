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

        elif params.dim == 3:
            # load pre-generated training data
            self.train_dataset = BrainLoadingDataset(train_data_path, "train", params.dim, data_pair=params.data_pair,
                                                     atlas_path=params.atlas_path)

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
        self.val_dataset = BrainLoadingDataset(val_data_path, "val", params.dim,
                                               data_pair=params.data_pair, atlas_path=params.atlas_path)

        self.val_dataloader = ptdata.DataLoader(self.val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=args.cuda
                                                )

        self.test_dataset = BrainLoadingDataset(test_data_path, "test", params.dim,
                                                data_pair=params.data_pair, atlas_path=params.atlas_path)

        self.test_dataloader = ptdata.DataLoader(self.test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=args.cuda
                                                 )




""" Base Dataset """


class _BaseDataset(ptdata.Dataset):
    def __init__(self, data_path, run, dim, slice_range=(70, 90)):
        super(_BaseDataset, self).__init__()
        self.data_path = data_path
        self.run = run  # "train", "val" or "test"
        self.dim = dim
        self.subject_list = sorted(os.listdir(self.data_path))

        self.slice_range = slice_range

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
    def _normalise_intensity(data_dict):
        # intensity normalisation to [0, 1]
        for name in ["target_original", "source"]:
            data_dict[name] = normalise_intensity(data_dict[name],
                                                  min_out=0., max_out=1.,
                                                  mode="minmax", clip=True)
        return data_dict

    def _load_2d(self, data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            if name == "dvf_gt":
                if self.run == "train":
                    continue
                # dvf.yaml is saved in shape (H, W, N, 2) -> (N, 2, H, W)
                data_dict[name] = load_nifti(data_path).transpose(2, 3, 0, 1)

            else:
                # image is saved in shape (H, W, N) ->  (N, H, W)
                data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)

        # randomly select a slice for training
        if self.run == "train":
            z = random.randint(self.slice_range[0], self.slice_range[1])
            slicer = slice(z, z + 1)  # use slicer to keep dim
            for name, data in data_dict.items():
                data_dict[name] = data[slicer, ...]  # (1, H, W)
        return data_dict

    def _load_3d(self, data_path_dict):
        data_dict = dict()
        for name, data_path in data_path_dict.items():
            if name == "dvf_gt":
                # skip loading ground truth DVF for training
                if self.run == "train":
                    continue
                # dvf_gt is saved in shape (H, W, D, 3) -> (ch=1, 3, H, W, D)
                data_dict[name] = load_nifti(data_path).transpose(3, 0, 1, 2)[np.newaxis, ...]

            else:
                # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
                data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
        return data_dict

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and process"""
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)




""" Loading pre-generated data Datasets """


class BrainLoadingDataset(_BaseDataset):
    """
    Dataset that loads saved generated & pre-processed data
    """
    def __init__(self, data_path, run, dim, data_pair,
                 slice_range=(70, 90), atlas_path=None):
        super(BrainLoadingDataset, self).__init__(data_path, run, dim, slice_range=slice_range)
        self.data_pair = data_pair
        self.atlas_path = atlas_path

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        data_path_dict = dict()
        if self.data_pair == "intra":  # intra-subject
            subj_id = self.subject_list[index]
            for name in ["target", "source", "target_original", "roi_mask", "dvf_gt", "cor_seg", "subcor_seg"]:
                data_path_dict[name] = f"{self.data_path}/{subj_id}/{name}.nii.gz"

        else:  # inter-subject
            if self.data_pair == "inter_random":
                tar_subj_id = self.subject_list[index]
                if self.run == "train":
                    # randomly choose source subject from other subjects
                    src_subj_pool = self.subject_list.copy()
                    src_subj_pool.remove(tar_subj_id)
                    src_subj_id = random.choice(src_subj_pool)

                else:
                    # for val/test, fixed pairing with the next subject on the list (looping)
                    if index < len(self.subject_list) - 1:
                        src_subj_id = self.subject_list[index+1]
                    else:
                        src_subj_id = self.subject_list[0]

                # set target data paths for intra-subject
                data_path_dict["target"] = f"{self.data_path}/{tar_subj_id}/target.nii.gz"
                data_path_dict["target_cor_seg"] = f"{self.data_path}/{tar_subj_id}/cor_seg.nii.gz"
                data_path_dict["target_subcor_seg"] = f"{self.data_path}/{tar_subj_id}/subcor_seg.nii.gz"

            elif self.data_pair == "inter_atlas":
                assert self.atlas_path is not None, "Atlas path not given."
                # set target data paths for inter-subject
                data_path_dict["target"] = f"{self.atlas_path}/target.nii.gz"
                data_path_dict["target_cor_seg"] = f"{self.atlas_path}/cor_seg.nii.gz"
                data_path_dict["target_subcor_seg"] = f"{self.atlas_path}/subcor_seg.nii.gz"

                src_subj_id = self.subject_list[index]

            else:
                raise ValueError(f"Data pairing setting ({self.data_pair}) not recognised.")

            # set source data paths
            data_path_dict["source"] = f"{self.data_path}/{src_subj_id}/source.nii.gz"
            data_path_dict["source_cor_seg"] = f"{self.data_path}/{src_subj_id}/cor_seg.nii.gz"
            data_path_dict["source_subcor_seg"] = f"{self.data_path}/{src_subj_id}/subcor_seg.nii.gz"

            # set brain mask path
            data_path_dict["roi_mask"] = f"{self.data_path}/{src_subj_id}/roi_mask.nii.gz"

        return data_path_dict

    def __getitem__(self, index):
        data_path_dict = self._set_path(index)
        data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)  # load 2d/3d
        return self._to_tensor(data_dict)



""" Synthesis datasets """


class _SynthDataset(_BaseDataset):
    def __init__(self,
                 data_path,
                 run,
                 dim,
                 slice_range=(70, 90),
                 sigma=8,
                 cps=10,
                 disp_max=1.,
                 crop_size=(192, 192, 192),
                 device=torch.device('cpu')  # ??
                 ):
        """
        Loading, processing and synthesising transformation
        Args:
            sigma: (int, float or tuple) sigma of the Gaussian smoothing filter
            cps: (int, float or tuple) Control point spacing
            disp_max: (int, float or tuple) Maximum displacement of the control points
            crop_size: (int or tuple) Size of the image to crop into
            device: (torch.device)
        """
        super(_SynthDataset, self).__init__(data_path, run, dim, slice_range=slice_range)

        self.crop_size = crop_size  # todo: dimension check to enable integer argument
        self.device = device

        # elastic parameters
        self.sigma = sigma
        self.cps = cps
        self.disp_max = disp_max

        # Gaussian smoothing filter for random transformation generation
        self.smooth_filter = GaussianFilter(dim=self.dim, sigma=self.sigma)

    def _set_path(self, index):
        """ Set the paths of data files to load """
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
        data_path_dict = self._set_path(index)
        data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)  # load 2d/3d
        data_dict = self._crop_and_pad(data_dict, self.crop_size)
        data_dict = self._normalise_intensity(data_dict)
        data_dict = self._synthesis(data_dict)
        return self._to_tensor(data_dict)


class BratsSynthDataset(_SynthDataset):
    def __init__(self, *args, **kwargs):
        super(BratsSynthDataset, self).__init__(*args, **kwargs)

    def _set_path(self, index):
        subj_id = self.subject_list[index]
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{self.data_path}/{subj_id}/{subj_id}_t1.nii.gz"
        data_path_dict["source"] = f"{self.data_path}/{subj_id}/{subj_id}_t2.nii.gz"
        data_path_dict["roi_mask"] = f"{self.data_path}/{subj_id}/{subj_id}_brainmask.nii.gz"
        return data_path_dict


class IXISynthDataset(_SynthDataset):
    def __init__(self, *args, **kwargs):
        super(IXISynthDataset, self).__init__(*args, **kwargs)

    def _set_path(self, index):
        subj_id = self.subject_list[index]
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{self.data_path}/{subj_id}/T1-brain.nii.gz"
        data_path_dict["source"] = f"{self.data_path}/{subj_id}/T2-brain.nii.gz"
        data_path_dict["roi_mask"] = f"{self.data_path}/{subj_id}/T1-brain_mask.nii.gz"
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

    def _set_path(self, index):
        subj_id = self.subject_list[index]
        data_path_dict = dict()
        data_path_dict["target_original"] = f"{self.data_path}/{subj_id}/T1_brain.nii.gz"
        data_path_dict["source"] = f"{self.data_path}/{subj_id}/T2_brain.nii.gz"
        data_path_dict["roi_mask"] = f"{self.data_path}/{subj_id}/T1_brain_mask.nii.gz"

        # structural segmentation maps
        data_path_dict["cor_seg"] = f"{self.data_path}/{subj_id}/fsl_cortical_seg.nii.gz"
        data_path_dict["subcor_seg"] = f"{self.data_path}/{subj_id}/fsl_all_fast_firstseg.nii.gz"
        return data_path_dict
