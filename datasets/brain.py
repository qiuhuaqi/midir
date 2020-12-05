import random
from datasets.base_dataset import _BaseDataset
from datasets.utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor


class BrainInterSubject3DTrain(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 modality='t1t1',
                 atlas_path=None):
        super(BrainInterSubject3DTrain, self).__init__(data_dir_path)
        self.crop_size = crop_size
        self.atlas_path = atlas_path
        self.modality = modality

        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source']

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['target'] = f'{self.tar_subj_path}/T1_brain.nii.gz'

        if self.modality == 't1t1':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T2_brain.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.image_keys)
        return _to_tensor(data_dict)


class BrainInterSubject3DEval(BrainInterSubject3DTrain):
    def __init__(self, *args, **kwargs):
        super(BrainInterSubject3DEval, self).__init__(*args, **kwargs)
        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source', 'target_original']

    def _set_path(self, index):
        super(BrainInterSubject3DEval, self)._set_path(index)

        # T1w image of source subject (for visualisation)
        # TODO: the name 'target_original' is only a suitable for intra-subject setting
        self.data_path_dict['target_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'

        # segmentation
        self.data_path_dict['target_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        self.data_path_dict['source_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

