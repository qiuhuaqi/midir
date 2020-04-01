"""Datasets written in compliance with Pytorch DataLoader interface"""
import os
import os.path as path
import random
import numpy as np
from glob import glob
import nibabel as nib
import torch
import torch.utils.data as ptdata

from data.utils import synthesis_elastic_deformation, CenterCrop
from utils.image import normalise_intensity

"""
Data object:
- Construct Datasets and Dataloaders
- Standardize data interface
"""

class Data(object):
    def __init__(self, *args):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.args = args[0]
        self.params = args[1]

    @staticmethod
    def _worker_init_fn(worker_id):
        """Callback function passed to DataLoader to initialise the workers"""
        # # generate a random sequence of seeds for the workers
        # print(f"Random state before generating the random seed: {random.getstate()}")

        random_seed = random.randint(0, 2 ** 32 - 1)

        # ##debug
        # print(f"Random state after generating the random seed: {random.getstate()}")
        # print(f"Random seed for worker {worker_id} is: {random_seed}")
        # ##

        np.random.seed(random_seed)

    def use_brain(self):
        # parse tuple JSON params
        self.params.slice_range = (self.params.slice_start, self.params.slice_end)
        self.params.disp_range = (self.params.disp_min, self.params.disp_max)

        # training
        self.train_dataset = Brats2D(self.params.data_path,
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
                                                  worker_init_fn=self._worker_init_fn  # todo: fix random seeding
                                                  )


        # validation
        self.val_dataset = Brats2D(self.params.data_path,
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
        self.test_dataset = Brats2D(self.params.data_path,
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

class Brats2D(ptdata.Dataset):
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
                             f"/{run}_crop{crop_size}_sigma{sigma}_cps{cps}_dispRange{disp_range[0]}-{disp_range[1]}_sliceRange{slice_range[0]}-{slice_range[1]}"
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
        # load the original data
        subject = self.subject_list[index]

        t1_path = path.join(self.data_path, subject, f"{subject}_t1.nii.gz")
        t2_path = path.join(self.data_path, subject, f"{subject}_t2.nii.gz")
        brain_mask_path = path.join(self.data_path, subject, f"{subject}_brainmask.nii.gz")

        # load in T1 &/ T2 image and brain mask, transpose to (NxHxW)
        target_original = nib.load(t1_path).get_data().transpose(2, 0, 1).astype("float")
        source = nib.load(t2_path).get_data().transpose(2, 0, 1).astype("float")
        brain_mask = nib.load(brain_mask_path).get_data().transpose(2, 0, 1).astype("float")

        """Different processing for train/generate/eval"""
        ## Training data: random one slice in range, crop, minmax & meanstd normalisation
        if self.run == "train":
            # taking a random slice each time
            z = random.randint(self.slice_range[0], self.slice_range[1])
            target_original = target_original[np.newaxis, z, ...]  # (1xHxW)
            source = source[np.newaxis, z, ...]  # (1xHxW)
            brain_mask = brain_mask[np.newaxis, z, ...]  #(1xHxW)

            # intensity normalisation 1: minmax
            target_original, source = map(lambda x:
                                          normalise_intensity(x, mode="minmax", clip=True),
                                          [target_original, source])  # (1, H, W)


            # generate synthesised DVF and deformed T1 image
            # image size: (1, H, W)
            # dvf size: (1, 2, H, W), in number of pixels
            target, dvf, _ = synthesis_elastic_deformation(target_original,
                                                           brain_mask,
                                                           sigma=self.sigma,
                                                           cps=self.cps,
                                                           disp_range=self.disp_range
                                                           )
            # cropping
            target, source, target_original, brain_mask = map(self.cropper,
                                                              [target, source, target_original, brain_mask])
            dvf_crop = []
            for dim in range(dvf.shape[1]):
                dvf_crop += [self.cropper(dvf[:, dim, :, :])]  # (N, H, W)
            dvf = np.array(dvf_crop).transpose((1, 0, 2, 3))  # (N, 2, H, W)

            # intensity normalisation 2: meanstd
            target, target_original, source = map(lambda x: normalise_intensity(x, mode="meanstd"),
                                                  [target, target_original, source])  # (1, H, W)


        ## Generate val/test data: all slices in range, crop, minmax normalisation
        elif self.run == "generate":
            # take a range of slices
            target_original = target_original[self.slice_range[0]: self.slice_range[1], ...]  # (N, H, W)
            source = source[self.slice_range[0]: self.slice_range[1], ...]  # (N, H, W)
            brain_mask = brain_mask[self.slice_range[0]: self.slice_range[1], ...]  # (N, H, W)

            # min-max intensity normalisation
            target_original, source = map(lambda x: normalise_intensity(x, mode="minmax", clip=True),
                                          [target_original, source])  # (N, H, W)

            # generate synthesised DVF and deformed T1 image
            # image shape: (N, H, W)
            # dvf shape: (N, 2, H, W), in number of pixels
            target, dvf, mask_bbox_mask = synthesis_elastic_deformation(target_original,
                                                                        brain_mask,
                                                                        sigma=self.sigma,
                                                                        cps=self.cps,
                                                                        disp_range=self.disp_range
                                                                        )
            # cropping
            target, source, target_original, brain_mask = map(self.cropper,
                                                              [target, source, target_original, brain_mask])
            dvf_crop = []
            for dim in range(dvf.shape[1]):
                dvf_crop += [self.cropper(dvf[:, dim, :, :])]  # (N, H, W)
            dvf = np.array(dvf_crop).transpose(1, 0, 2, 3)  # (N, 2, H, W)


        ## Validation/testing: load in saved target (T1-deformed, minmax normalised) and DVF
        elif self.run == "val" or self.run == "test":
            t1_deformed_path = path.join(self.data_path, subject, f"{subject}_t1_deformed.nii.gz")
            t1_deformed = nib.load(t1_deformed_path).get_data().transpose(2, 0, 1)  # (N, H, W)
            target = t1_deformed

            dvf_path = glob(path.join(self.data_path, subject, "*dvf*.nii.gz"))[0]
            dvf = nib.load(dvf_path).get_data().transpose(2, 3, 0, 1)  # (N, 2, H, W)
            assert dvf.shape[1] == 2, "Loaded DVF shape dim 1 is not 2."

        else:
            raise ValueError("Dataset run state not specified.")

        ## All cast to float32 Tensor
        target, source, target_original, brain_mask, dvf = map(lambda x: torch.from_numpy(x).float(),
                                                               [target, source, target_original, brain_mask, dvf])

        # shape of images: (N, H, W), N = 1 for train, = num_slices for generate/val/test
        # shape of dvf: (N, 2, H, W)
        return target, source, target_original, brain_mask, dvf

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

