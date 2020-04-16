"""Datasets written in compliance with Pytorch DataLoader interface"""
import os
import os.path as path
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as ptdata

from data.transforms import CenterCrop
from data.elastics import synthesis_elastic_deformation
from utils.image import normalise_intensity
from utils.image_io import load_nifti

"""
Data object:
- Construct Datasets and Dataloaders
- Standardize data interface
"""

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



class Brain2dData(object):
    def __init__(self, args, params):

        self.args = args
        self.params = params

        # parse tuple JSON params
        self.params.slice_range = (self.params.slice_start, self.params.slice_end)
        self.params.disp_range = (self.params.disp_min, self.params.disp_max)

        # training
        self.train_dataset = Brain2dDataset(self.params.data_path,
                                            run="train",
                                            slice_range=self.params.slice_range,
                                            sigma=self.params.sigma,
                                            cps=self.params.elastic_cps,
                                            disp_range=self.params.disp_range,
                                            crop_size=self.params.crop_size
                                            )

        self.train_dataloader = ptdata.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=self.args.cuda,
                                                  worker_init_fn=worker_init_fn  # todo: fix random seeding
                                                  )


        # validation
        self.val_dataset = Brain2dDataset(self.params.data_path,
                                          run="val",
                                          slice_range=self.params.slice_range,
                                          sigma=self.params.sigma,
                                          cps=self.params.elastic_cps,
                                          disp_range=self.params.disp_range,
                                          crop_size=self.params.crop_size
                                          )

        self.val_dataloader = ptdata.DataLoader(self.val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=self.args.num_workers,
                                                pin_memory=self.args.cuda
                                                )


        # testing
        self.test_dataset = Brain2dDataset(self.params.data_path,
                                           run="test",
                                           slice_range=self.params.slice_range,
                                           sigma=self.params.sigma,
                                           cps=self.params.elastic_cps,
                                           disp_range=self.params.disp_range,
                                           crop_size=self.params.crop_size
                                           )

        self.test_dataloader = ptdata.DataLoader(self.test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=self.args.cuda
                                                 )


"""
Datasets
"""
class Brain2dDataset(ptdata.Dataset):
    def __init__(self,
                 data_path,
                 run=None,
                 slice_range=(70, 90),
                 sigma=8,
                 cps=10,
                 disp_range=(0, 3),
                 crop_size=192
                 ):
        super().__init__()

        # set up train/val/test data path
        self.run = run
        if self.run == "train":
            self.data_path = data_path + "/train"

        elif self.run == "generate":
            self.data_path = data_path

        elif self.run == "val" or self.run == "test":
            self.data_path = data_path + \
                             f"/{run}" \
                             f"_crop{crop_size}" \
                             f"_sigma{sigma}" \
                             f"_cps{cps}" \
                             f"_dispRange{disp_range[0]}-{disp_range[1]}" \
                             f"_sliceRange{slice_range[0]}-{slice_range[1]}"
        else:
            raise ValueError("Dataset run state not specified.")
        assert path.exists(self.data_path), f"Data path does not exist: \n{self.data_path}"

        self.subject_list = sorted(os.listdir(self.data_path))

        # elastic parameters
        self.sigma = sigma
        self.cps = cps
        self.disp_range = disp_range
        self.slice_range = slice_range

        # cropper
        self.cropper = CenterCrop(crop_size)

    def __getitem__(self, index):
        data_dict = {}

        subject_id = self.subject_list[index]

        # load in T1 &/ T2 image and brain mask, transpose to (N, *(dims))
        t1_path = f"{self.data_path}/{subject_id}/{subject_id}_t1.nii.gz"
        t2_path = f"{self.data_path}/{subject_id}/{subject_id}_t2.nii.gz"
        brain_mask_path = f"{self.data_path}/{subject_id}/{subject_id}_brainmask.nii.gz"

        data_dict["target_original"] = load_nifti(t1_path).transpose(2, 0, 1)
        data_dict["source"] = load_nifti(t2_path).transpose(2, 0, 1)
        data_dict["roi_mask"] = load_nifti(brain_mask_path).transpose(2, 0, 1)

        if self.run == "train":
            """
            Training data
            """
            # taking a random slice each time
            z = random.randint(self.slice_range[0], self.slice_range[1])
            for name, data in data_dict.items():
                data_dict[name] = data[np.newaxis, z, ...]  # (1xHxW)

            # intensity normalisation
            for name in ["target_original", "source"]:
                data_dict[name] = normalise_intensity(data_dict[name], mode="minmax", clip=True)  # (1, H, W)

            # generate synthesised DVF and deformed T1 image
            # image and mask shape: (1, H, W)
            # dvf_gt shape: (1, 2, H, W), in number of pixels
            data_dict["target"], data_dict["dvf_gt"] = synthesis_elastic_deformation(data_dict["target_original"],
                                                                                     data_dict["roi_mask"],
                                                                                     sigma=self.sigma,
                                                                                     cps=self.cps,
                                                                                     disp_range=self.disp_range
                                                                                     )
            # cropping
            for name in ["target", "source", "target_original", "roi_mask"]:
                data_dict[name] = self.cropper(data_dict[name])

            dvf_crop = []
            for d in range(data_dict["dvf_gt"].shape[1]):
                dvf_crop += [self.cropper(data_dict["dvf_gt"][:, d , :, :])]  # (N, H, W)
            data_dict["dvf_gt"] = np.swapaxes(np.array(dvf_crop), 0, 1) # (N, dim, H, W)

        elif self.run == "generate":
            """
            Generate val/test data
            """
            # take a range of slices
            for name, data in data_dict.items():
                data_dict[name] = data[self.slice_range[0]: self.slice_range[1], ...]  # (N, H, W)

            # intensity normalisation
            for name in ["target_original", "source"]:
                data_dict[name] = normalise_intensity(data_dict[name], mode="minmax", clip=True)  # (1, H, W)

            # generate synthesised DVF and deformed T1 image
            # image shape: (N, H, W)
            # dvf_gt shape: (N, 2, H, W), in number of pixels
            data_dict["target"], data_dict["dvf_gt"] = synthesis_elastic_deformation(data_dict["target_original"],
                                                                                     data_dict["roi_mask"],
                                                                                     sigma=self.sigma,
                                                                                     cps=self.cps,
                                                                                     disp_range=self.disp_range
                                                                                     )
            # cropping
            for name in ["target", "source", "target_original", "brain_mask"]:
                data_dict[name] = self.cropper(data_dict[name])

            dvf_crop = []
            for d in range(data_dict["dvf_gt"].shape[1]):
                dvf_crop += [self.cropper(data_dict["dvf_gt"][:, d , :, :])]  # [(N, H, W)]
            data_dict["dvf_gt"] = np.swapaxes(np.array(dvf_crop), 0, 1) # (N, dim, H, W)


        elif self.run == "val" or self.run == "test":
            """
            Load saved val/test data
            """
            t1_deformed_path = f"{self.data_path}/{subject_id}/{subject_id}_t1_deformed.nii.gz"
            data_dict["target"] = load_nifti(t1_deformed_path).transpose(2, 0, 1)  # (N, H, W)

            dvf_path = glob(f"{self.data_path}/{subject_id}/*dvf*.nii.gz")[0]
            data_dict["dvf_gt"] = load_nifti(dvf_path).transpose(2, 3, 0, 1)  # (N, 2, H, W)
        else:
            raise ValueError("Dataset run state not specified.")
        """"""

        # Cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()

        # shape of images & mask: (N, *(dims,)),
        # N = 1 for train, = num_slices for generate/val/test
        # shape of dvf_gt: (N, dim, (*dims,))
        return data_dict

    def __len__(self):
        return len(self.subject_list)



class IXI2D(ptdata.Dataset):
    """
    Load IXI data for 2D registration
    """

    def __init__(self, data_path, num_slices=50, augment=False, transform=None):
        super(IXI2D, self).__init__()

        self.data_path = data_path
        self.num_slices = num_slices
        self.augment = augment
        self.transform = transform

        self.subject_list = None

    def __getitem__(self, index):
        target = None
        source = None
        return target, source

    def __len__(self):
        return len(self.subject_list)

