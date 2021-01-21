from datasets.base_dataset import _BaseDataset
from datasets.utils import _load2d, _crop_and_pad, _normalise_intensity, _to_tensor, _magic_slicer


class CardiacMR2D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 evaluate=False,
                 slice_range=None,
                 slicing=None,
                 crop_size=(172, 172),
                 ):
        super(CardiacMR2D, self).__init__(data_dir_path)

        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source']

        self.evaluate = evaluate
        self.slice_range = slice_range
        self.slicing = slicing
        self.crop_size = crop_size

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['target'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['source'] = f'{self.subj_path}/sa_ES.nii.gz'
        if self.evaluate:
            self.data_path_dict['target_original'] = self.data_path_dict['source']
            self.data_path_dict['target_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
            self.data_path_dict['source_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.image_keys)
        return _to_tensor(data_dict)

import omegaconf
x = omegaconf.listconfig.ListConfig