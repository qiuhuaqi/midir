import datetime
import os
import os.path as path
import random
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data

class CardiacMR_2D_UKBB(data.Dataset):
    """
    Training dataset class for UK Biobank dataset
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
            if path.exists(path.join(data_path, subj_dir, seq+'.nii.gz')) \
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

        # define source and target images: source is a sequence of images
        target = image_ed[np.newaxis, :, :]  # extend dim to (1, H, W)
        if image.shape[0] > self.seq_length:
            # if the required sequence is longer than seq_length, take a sequence out
            start_frame_idx = random.randint(0, image.shape[0] - self.seq_length)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
        else:
            # if the required sequence is shorter than seq_length, use the whole sequence
            source = image[1:, :, :]  # (T-1, H, W)

        # transformation functions expect input shape (N, H, W)
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source

    def __len__(self):
        return len(self.dir_list)

class SupervisedWarpedCardiacMR2DUKBB(CardiacMR_2D_UKBB):
    def __init__(self, data_path, seq='sa', crop_size=160, seq_length=30, augment=False, transform=None):
        super(SupervisedWarpedCardiacMR2DUKBB, self).__init__(data_path, seq=seq, seq_length=seq_length,
                                                              augment=augment,
                                                              transform=transform)
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        Outputs the warped source images and original source images as inputs to the network

        Args:
            index:

        Returns:
            target: (Tensor of size (seq_length, H, W)) warped source image sequence using DVF
            source: (Tensor of size (seq_length, H, W)) source image sequence
            dvf: (Tensor of size (seq_length, 2, H, W)) DVF sequence, from target to source
        """
        subj_dir = os.path.join(self.data_path, self.dir_list[index])

        # load in the image and dvf sequences
        image_raw_path = os.path.join(subj_dir, self.seq + '.nii.gz')
        image_stack_seq = nib.load(image_raw_path).get_data()  # (H, W, N, T)
        target_raw_path = os.path.join(subj_dir, "dvf_warped_{seq}.nii.gz".format(seq=self.seq))
        target_stack_seq = nib.load(target_raw_path).get_data()  # (H, W, N, T-1)
        dvf_raw_path = os.path.join(subj_dir, 'dvf.nii.gz')
        dvf_stack_seq = nib.load(dvf_raw_path).get_data()  # (H, W, N, T-1, 2)

        # random select a z-axis slice
        if self.seq == 'sa':
            slice_num = random.randint(0, image_stack_seq.shape[-2] - 1)
        else:
            slice_num = 0

        # select the slice and transpose
        image_seq = image_stack_seq[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)  # (T, H, W)
        target_seq = target_stack_seq[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)  # (T-1, H, W)
        dvf_seq = dvf_stack_seq[:, :, slice_num, :, :].transpose(2, 3, 0, 1)  # (T-1, 2, H, W)

        # define source and target images:
        if image_seq.shape[0] > self.seq_length:
            start_frame_idx = random.randint(1, image_seq.shape[0] - self.seq_length + 1)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image_seq[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
            target = target_seq[(start_frame_idx-1):(end_frame_idx-1), :, :]  # (seq_length, H, W)
            dvf = dvf_seq[(start_frame_idx-1):(end_frame_idx-1), :, :, :]  # (seq_length, 2, H, W)
        else:
            # if the sequence is shorter than seq_length, use the whole sequence
            source = image_seq[1:, :, :]  # (T-1, H, W)
            target = target_seq  # (T-1, H, W)
            dvf = dvf_seq  # (T-1, 2, H, W)

        # normalise dvf coordinates to [-1, 1] and cast to Pytorch tensor
        dvf = 2 * dvf / self.crop_size
        dvf = torch.from_numpy(dvf)

        # apply transformation to images
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source, dvf

class SupervisedOriginalCardiacMR2DUKBB(CardiacMR_2D_UKBB):
    def __init__(self, data_path, seq='sa', crop_size=160, seq_length=30, augment=False, transform=None):
        super(SupervisedOriginalCardiacMR2DUKBB, self).__init__(data_path, seq=seq, seq_length=seq_length,
                                                                augment=augment,
                                                                transform=transform)
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        Outputs the original target and source as inputs to the network

        Args:
            index:

        Returns:
            target: (Tensor of size (1, H, W)) target image
            source: (Tensor of size (seq_length, H, W)) source image sequence
            dvf: (Tensor of size (seq_length, 2, H, W)) DVF sequence, from target to source
        """
        subj_dir = os.path.join(self.data_path, self.dir_list[index])

        # load in the image and dvf sequences
        image_raw_path = os.path.join(subj_dir, self.seq + '.nii.gz')
        image_stack_seq = nib.load(image_raw_path).get_data()  # (H, W, N, T)
        image_ed_path = os.path.join(subj_dir, self.seq + '_ED.nii.gz')
        image_ed = nib.load(image_ed_path).get_data()  # (H, W, N)
        dvf_raw_path = os.path.join(subj_dir, 'dvf.nii.gz')
        dvf_stack_seq = nib.load(dvf_raw_path).get_data()  # (H, W, N, T-1, 2)

        # random select a z-axis slice
        if self.seq == 'sa':
            slice_num = random.randint(0, image_stack_seq.shape[-2] - 1)
        else:
            slice_num = 0

        # select the slice and transpose
        image_seq = image_stack_seq[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)  # (T, H, W)
        image_ed = image_ed[:, :, slice_num]  # (H, W)
        dvf_seq = dvf_stack_seq[:, :, slice_num, :, :].transpose(2, 3, 0, 1)  # (T-1, 2, H, W)

        # define source and target images: source is a sequence of images
        target = image_ed[np.newaxis, :, :]  # extend dim to (1, H, W)

        if image_seq.shape[0] > self.seq_length:
            # if the required sequence is longer than seq_length, take a sequence out
            start_frame_idx = random.randint(1, image_seq.shape[0] - self.seq_length + 1)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image_seq[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
            dvf = dvf_seq[(start_frame_idx-1):(end_frame_idx-1), :, :, :]  # (seq_length, 2, H, W)
        else:
            # if the required sequence is shorter than seq_length, use the whole sequence
            source = image_seq[1:, :, :]  # (T-1, H, W)
            dvf = dvf_seq  # (T-1, 2, H, W)

        # normalise dvf coordinates to [-1, 1] and cast to Pytorch tensor
        dvf = 2 * dvf / self.crop_size
        dvf = torch.from_numpy(dvf)

        # apply transformation to images
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)

        return target, source, dvf

class CardiacMR_2D_Eval_UKBB(data.Dataset):
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
        label_path_ed = os.path.join(self.data_path, self.dir_list[index], '{}_'.format(self.label_prefix) + self.seq + '_ED.nii.gz')
        label_path_es = os.path.join(self.data_path, self.dir_list[index], '{}_'.format(self.label_prefix) + self.seq + '_ES.nii.gz')

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


class CardiacMR_2D_Inference_UKBB(data.Dataset):
    """Inference dataset which loops over frames of one subject"""
    def __init__(self, data_path, seq='sa', transform=None):
        """data_path is the path to the directory containing the NIFTI files"""
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
