"""Datasets written in compliance with Pytorch DataLoader interface"""
import datetime
import os
import os.path as path
import random
import numpy as np
from glob import glob
import nibabel as nib
import torch
import torch.utils.data as ptdata

from data.dataset_utils import CenterCrop, Normalise, ToTensor
import torchvision.transforms as tvtransforms

"""
Data object:
- Construct Datasets and Dataloaders
- Standarlise data interface
"""


class Data():
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
        # generate a random sequence of seeds for the workers
        np.random.seed(random.randint(0, 2 ** 32 - 1))

    def use_ukbb_cardiac(self):
        """
        create the dataloaders according to arguments and parameters
        """

        self.data_name = "UK Biobank cardiac"

        self.train_dataset = CardiacMR_2D_UKBB(self.params.train_data_path,
                                               seq=self.params.seq,
                                               seq_length=self.params.seq_length,
                                               augment=self.params.augment,
                                               transform=tvtransforms.Compose([
                                                   CenterCrop(self.params.crop_size),
                                                   Normalise(),
                                                   ToTensor()
                                               ]))

        self.train_dataloader = ptdata.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=self.args.cuda)

        self.val_dataset = CardiacMR_2D_Eval_UKBB(self.params.val_data_path,
                                                  seq=self.params.seq,
                                                  augment=self.params.augment,
                                                  label_prefix=self.params.label_prefix,
                                                  transform=tvtransforms.Compose([
                                                      CenterCrop(self.params.crop_size),
                                                      Normalise(),
                                                      ToTensor()]),
                                                  label_transform=tvtransforms.Compose([
                                                      CenterCrop(self.params.crop_size),
                                                      ToTensor()])
                                                  )

        self.val_dataloader = ptdata.DataLoader(self.val_dataset,
                                                batch_size=self.params.batch_size,
                                                shuffle=False,
                                                num_workers=self.args.num_workers,
                                                pin_memory=self.args.cuda)

        self.test_dataset = CardiacMR_2D_Eval_UKBB(self.params.eval_data_path,
                                                   seq=self.params.seq,
                                                   augment=self.params.augment,
                                                   label_prefix=self.params.label_prefix,
                                                   transform=tvtransforms.Compose([
                                                       CenterCrop(self.params.crop_size),
                                                       Normalise(),
                                                       ToTensor()]),
                                                   label_transform=tvtransforms.Compose([
                                                       CenterCrop(self.params.crop_size),
                                                       ToTensor()])
                                                   )

        self.test_dataloader = ptdata.DataLoader(self.test_dataset,
                                                 batch_size=self.params.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=self.args.cuda)

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
                                                  shuffle=True,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=self.args.cuda,
                                                  worker_init_fn=self._worker_init_fn
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


########################################################
# Datasets
########################################################
"""
Brain Datasets
"""
from utils.image_utils import synthesis_elastic_deformation


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

        # intensity normalisation
        self.normaliser_minmax = Normalise(mode='minmax')
        self.normaliser_meanstd = Normalise(mode='meanstd')

    def __getitem__(self, index):
        # load the original data
        subject = self.subject_list[index]

        t1_path = path.join(self.data_path, subject, f"{subject}_t1.nii.gz")
        t2_path = path.join(self.data_path, subject, f"{subject}_t2.nii.gz")
        brain_mask_path = path.join(self.data_path, subject, f"{subject}_brainmask.nii.gz")

        # load in T1 &/ T2 image and brain mask, transpose to (NxHxW)
        ## mono-modal or multi-modal
        target_original = nib.load(t1_path).get_data().transpose(2, 0, 1).astype("float")
        source = nib.load(t2_path).get_data().transpose(2, 0, 1).astype("float")
        brain_mask = nib.load(brain_mask_path).get_data().transpose(2, 0, 1).astype("float")

        """Different processing for train/generate/eval"""
        if self.run == "train":
            """Training data: random one slice in range, crop, normalise"""
            # taking a random slice each time
            z = random.randint(self.slice_range[0], self.slice_range[1])
            target_original = target_original[np.newaxis, z, ...]  # (1xHxW)
            source = source[np.newaxis, z, ...]  # (1xHxW)
            brain_mask = brain_mask[np.newaxis, z, ...]  #(1xHxW)
            assert target_original.shape[0] == 1, "Dataset training: more than 1 slice taken from one subject"

            # generate synthesised DVF and deformed T1 image
            # dvf size: (Nx2xHxW), in number of pixels
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

            # intensity normalisation
            target, source, target_original = map(self.normaliser_minmax, [target, source, target_original])
            target, source, target_original = map(self.normaliser_meanstd, [target, source, target_original])

        elif self.run == "generate":
            """Generate val/test data: all slices in range, crop, no normalisation"""
            # take a range of slices
            target_original = target_original[self.slice_range[0]: self.slice_range[1], ...]  # (num_slices xHxW)
            source = source[self.slice_range[0]: self.slice_range[1], ...]  # (num_slices xHxW)
            brain_mask = brain_mask[self.slice_range[0]: self.slice_range[1], ...]  # (num_slices xHxW)

            # generate synthesised DVF and deformed T1 image
            # dvf size: (Nx2xHxW), in number of pixels
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

        elif self.run == "val" or self.run == "test":
            """Validation/testing: load in saved data and deformation"""
            t1_deformed_path = path.join(self.data_path, subject, f"{subject}_t1_deformed.nii.gz")
            t1_deformed = nib.load(t1_deformed_path).get_data().transpose(2, 0, 1)  # (N, H, W)
            target = t1_deformed

            dvf_path = glob(path.join(self.data_path, subject, "*dvf*.nii.gz"))[0]
            dvf = nib.load(dvf_path).get_data().transpose(2, 3, 0, 1)  # (N, 2, H, W)
            assert dvf.shape[1] == 2, "Loaded DVF shape dim 1 is not 2."
        else:
            raise ValueError("Dataset run state not specified.")

        # all cast to float32 Tensor
        # Shape (N, x, H, W), N=1 for traininig, N=number_slices for val/test
        target, source, target_original, brain_mask, dvf = map(lambda x: torch.from_numpy(x).float(),
                                                               [target, source, target_original, brain_mask, dvf])

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


"""
Cardiac Datasets
"""


class CardiacMR_2D(ptdata.Dataset):
    """
    Training dataset. Uses the first frame in a sequence as target.
    """

    def __init__(self, data_path, seq='sa', seq_length=20, augment=False, transform=None):
        # super(TrainDataset, self).__init__()
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.dir_list = [dir_ for dir_ in sorted(os.listdir(self.data_path))]
        self.seq = seq
        self.seq_length = seq_length
        self.augment = augment
        self.transform = transform

    def __getitem__(self, index):
        """
        Load and pre-process the input image.

        Args:
            index: index into the dir list

        Returns:
            target: target image, Tensor of size (1, H, W)
            source: source image sequence, Tensor of size (seq_length, H, W)

        """

        # update the seed to avoid workers sample the same augmentation parameters
        if self.augment:
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load nifti into array
        file_path = os.path.join(self.data_path, self.dir_list[index], self.seq + '.nii.gz')
        nim = nib.load(file_path)
        image_raw = nim.get_data()

        # random select a z-axis slice and transpose into (T, H, W)
        slice_num = random.randint(0, image_raw.shape[-2] - 1)
        image = image_raw[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)

        # define source and target images:
        #   target images are copies of the ED frame (extended later on GPU to save memory)
        target = image[np.newaxis, 0, :, :]  # (1, H, W)

        #   source images are a sequence of params.seq_length frames
        if image.shape[0] > self.seq_length:
            start_frame_idx = random.randint(0, image.shape[0] - self.seq_length)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
        else:  # if the sequence is shorter than seq_length, use the whole sequence
            print("Warning: data sequence is shorter than set sequence length")
            source = image[1:, :, :]  # (T-1, H, W)

        # transformation functions expect input shaped (N, H, W)
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source

    def __len__(self):
        return len(self.dir_list)


class CardiacMR_2D_UKBB(ptdata.Dataset):
    """
    Training class for UKBB. Loads the specific ED file as target.
    """

    def __init__(self, data_path, seq='sa', seq_length=30, augment=False, transform=None):
        # super(TrainDataset, self).__init__()
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq
        self.seq_length = seq_length
        self.augment = augment
        self.transform = transform

        self.dir_list = []
        for subj_dir in sorted(os.listdir(self.data_path)):
            if path.exists(path.join(data_path, subj_dir, seq + '.nii.gz')) \
                    and path.exists(path.join(data_path, subj_dir, seq + '_ED.nii.gz')):
                self.dir_list += [subj_dir]

    def __getitem__(self, index):
        """
        Load and pre-process the input image.

        Args:
            index: index into the dir list

        Returns:
            target: target image, Tensor of size (1, H, W)
            source: source image sequence, Tensor of size (seq_length, H, W)
        """

        # update the seed to avoid workers sample the same augmentation parameters
        # if self.augment:
        #     np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load nifti into array
        subj_dir = os.path.join(self.data_path, self.dir_list[index])
        image_seq_path = os.path.join(subj_dir, self.seq + '.nii.gz')
        image_raw = nib.load(image_seq_path).get_data()
        image_ed_path = os.path.join(subj_dir, self.seq + '_ED.nii.gz')
        image_ed = nib.load(image_ed_path).get_data()

        if self.seq == 'sa':
            # random select a z-axis slice and transpose into (seq_length, H, W)
            slice_num = random.randint(0, image_raw.shape[-2] - 1)
        else:
            slice_num = 0
        image = image_raw[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)
        image_ed = image_ed[:, :, slice_num]

        ## define source and target images:
        #   target images are copies of the ED frame (extended later in training code to make use of Pytorch view)
        target = image_ed[np.newaxis, :, :]  # extend dim to (1, H, W)

        #   source images are a sequence of params.seq_length frames
        if image.shape[0] > self.seq_length:
            start_frame_idx = random.randint(0, image.shape[0] - self.seq_length)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
        else:
            # if the sequence is shorter than seq_length, use the whole sequence
            source = image[1:, :, :]  # (T-1, H, W)

        # transformation functions expect input shape (N, H, W)
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source

    def __len__(self):
        return len(self.dir_list)


class CardiacMR_2D_Eval_UKBB(ptdata.Dataset):
    """Validation and evaluation for UKBB
    Fetches ED and ES frame images and segmentation labels"""

    def __init__(self, data_path, seq='sa', label_prefix='label', augment=False, transform=None, label_transform=None):
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq
        self.label_prefix = label_prefix
        self.augment = augment

        self.transform = transform
        self.label_transform = label_transform

        # check required data files
        self.dir_list = []
        for subj_dir in sorted(os.listdir(self.data_path)):
            if path.exists(path.join(data_path, subj_dir, seq + '_ES.nii.gz')) \
                    and path.exists(path.join(data_path, subj_dir, seq + '_ED.nii.gz')) \
                    and path.exists(path.join(data_path, subj_dir, '{}_'.format(label_prefix) + seq + '_ED.nii.gz')) \
                    and path.exists(path.join(data_path, subj_dir, '{}_'.format(label_prefix) + seq + '_ES.nii.gz')):
                self.dir_list += [subj_dir]

    def __getitem__(self, index):
        """
        Load and pre-process input image and label maps
        For now batch size is expected to be 1 and each batch contains
        images and labels for each subject at ED and ES (stacks)

        Args:
            index:

        Returns:
            image_ed, image_es, label_ed, label_es: Tensors of size (N, H, W)

        """
        # update the seed to avoid workers sample the same augmentation parameters
        if self.augment:
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load nifti into array
        image_path_ed = os.path.join(self.data_path, self.dir_list[index], self.seq + '_ED.nii.gz')
        image_path_es = os.path.join(self.data_path, self.dir_list[index], self.seq + '_ES.nii.gz')
        label_path_ed = os.path.join(self.data_path, self.dir_list[index],
                                     '{}_'.format(self.label_prefix) + self.seq + '_ED.nii.gz')
        label_path_es = os.path.join(self.data_path, self.dir_list[index],
                                     '{}_'.format(self.label_prefix) + self.seq + '_ES.nii.gz')

        # images and labels are in shape (H, W, N)
        image_ed = nib.load(image_path_ed).get_data()
        image_es = nib.load(image_path_es).get_data()
        label_ed = nib.load(label_path_ed).get_data()
        label_es = nib.load(label_path_es).get_data()

        # transpose into (N, H, W)
        image_ed = image_ed.transpose(2, 0, 1)
        image_es = image_es.transpose(2, 0, 1)
        label_ed = label_ed.transpose(2, 0, 1)
        label_es = label_es.transpose(2, 0, 1)

        # transformation functions expect input shaped (N, H, W)
        if self.transform:
            image_ed = self.transform(image_ed)
            image_es = self.transform(image_es)

        if self.label_transform:
            label_ed = self.label_transform(label_ed)
            label_es = self.label_transform(label_es)

        return image_ed, image_es, label_ed, label_es

    def __len__(self):
        return len(self.dir_list)


class CardiacMR_2D_Inference_UKBB(ptdata.Dataset):
    """Inference dataset, works with UKBB data or data with segmentation,
    loop over frames of one subject"""

    def __init__(self, data_path, seq='sa', transform=None):
        """data_path is the path to the direcotry containing the nifti files"""
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq

        self.transform = transform
        self.seq_length = None

        # load sequence image nifti
        file_path = os.path.join(self.data_path, self.seq + '.nii.gz')
        nim = nib.load(file_path)
        self.image_seq = nim.get_data()

        # pass sequence length to object handle
        self.seq_length = self.image_seq.shape[-1]

    def __getitem__(self, idx):
        """Return all slices of one frame in a sequence"""

        target = self.image_seq[:, :, :, 0].transpose(2, 0, 1)
        source = self.image_seq[:, :, :, idx].transpose(2, 0, 1)

        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source

    def __len__(self):
        return self.seq_length
