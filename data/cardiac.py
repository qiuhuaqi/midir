from data.base_dataset import _BaseDataset
from data.utils import _load2d, _crop_and_pad, _normalise_intensity, _to_tensor


class CardiacMR2DTrain(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size=(192, 192)):
        super(CardiacMR2DTrain, self).__init__(data_dir_path)
        self.crop_size = crop_size

        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source']

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'

        self.data_path_dict = dict()
        self.data_path_dict['target'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['source'] = f'{self.subj_path}/sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        # load one slice from each subject
        data_dict = _load2d(self.data_path_dict, random_slice=True)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.image_keys)
        return _to_tensor(data_dict)


class CardiacMR2DEval(CardiacMR2DTrain):
    def __init__(self, *args, **kwargs):
        super(CardiacMR2DEval, self).__init__(*args, **kwargs)

    def _set_path(self, index):
        super(CardiacMR2DEval, self)._set_path(index)
        # TODO: check the NIFTI file names
        self.data_path_dict['target_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
        self.data_path_dict['source_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'