import random

from data.base_datasets import _BaseDataset3D


class BrainInterSubject3DTrain(_BaseDataset3D):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 modality='mono',
                 atlas_path=None):
        super(BrainInterSubject3DTrain, self).__init__(data_dir_path)
        self.crop_size = crop_size
        self.atlas_path = atlas_path
        self.modality = modality

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict = dict()
        self.data_path_dict['target'] = f'{self.tar_subj_path}/T1_brain.nii.gz'
        if self.modality == 'mono':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 'multi':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T2_brain.nii.gz'

        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source']

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = self._load(self.data_path_dict)
        data_dict = self._crop_and_pad(data_dict, self.crop_size)
        data_dict = self._normalise_intensity(data_dict, self.image_keys)
        return self._to_tensor(data_dict)


class BrainInterSubject3DEval(BrainInterSubject3DTrain):
    def __init__(self, *args, **kwargs):
        super(BrainInterSubject3DEval, self).__init__(*args, **kwargs)
    
    def _set_path(self, index):
        super(BrainInterSubject3DEval, self)._set_path(index)

        # T1w image of source subject (for visualisation)
        # TODO: the name 'target_original' is only a suitable for intra-subject setting
        self.data_path_dict['target_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'

        # tells which data points are images (to normalise intensity)
        self.image_keys = ['target', 'source', 'target_original']

        # segmentation
        self.data_path_dict['target_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        self.data_path_dict['source_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'


# class BrainLoadingDataset(_BaseDataset):
#     """
#     (DEPRECATED)
#     Dataset that loads pre-processed/generated data
#     """
#     def __init__(self, data_dir, run, dim, data_pair,
#                  slice_range=(70, 90), atlas_path=None):
#         super(BrainLoadingDataset, self).__init__(data_dir, run, dim, slice_range=slice_range)
#
#         assert os.path.exists(data_dir), f"Data dir does not exist: {data_dir}"
#         self.data_pair = data_pair
#         self.atlas_path = atlas_path
#
#     def _set_path(self, index):
#         """ Set the paths of data files to load and the keys in data_dict"""
#         data_path_dict = dict()
#
#         # intra-subject mode
#         if self.data_pair == "intra":
#             subj_id = self.subject_list[index]
#             data_path_dict["target"] = f"{self.data_dir}/{subj_id}/T1w_synth.nii.gz"
#             data_path_dict["source"] = f"{self.data_dir}/{subj_id}/T2w.nii.gz"
#             data_path_dict["roi_mask"] = f"{self.data_dir}/{subj_id}/roi_mask.nii.gz"
#
#             if self.run == "eval":
#                 data_path_dict["target_original"] = f"{self.data_dir}/{subj_id}/T1w.nii.gz"
#                 data_path_dict["target_cor_seg"] = f"{self.data_dir}/{subj_id}/cor_seg_synth.nii.gz"
#                 data_path_dict["target_subcor_seg"] = f"{self.data_dir}/{subj_id}/subcor_seg_synth.nii.gz"
#                 data_path_dict["source_cor_seg"] = f"{self.data_dir}/{subj_id}/cor_seg.nii.gz"
#                 data_path_dict["source_subcor_seg"] = f"{self.data_dir}/{subj_id}/subcor_seg.nii.gz"
#                 data_path_dict["dvf_gt"] = f"{self.data_dir}/{subj_id}/dvf_gt.nii.gz"
#
#         # inter-subject mode (including intra-subject, randomly choose synthesised images)
#         elif self.data_pair == "inter":
#             tar_subj_id = self.subject_list[index]
#             src_subj_id = random.choice(self.subject_list)  # allows choosing the same subject (incl. intra)
#
#             # images
#             if self.run == "train" and random.choice([True, False]):
#                 # training: randomly choose to use synthesised image as target image
#                 data_path_dict["target"] = f"{self.data_dir}/{tar_subj_id}/T1w_synth.nii.gz"
#             else:
#                 data_path_dict["target"] = f"{self.data_dir}/{tar_subj_id}/T1w.nii.gz"
#
#             data_path_dict["source"] = f"{self.data_dir}/{src_subj_id}/T2w.nii.gz"
#             data_path_dict["roi_mask"] = f"{self.data_dir}/{tar_subj_id}/roi_mask.nii.gz"
#
#             if self.run == "eval":
#                 # T1w image of the other subject for error maps and RMSE
#                 data_path_dict["target_original"] = f"{self.data_dir}/{src_subj_id}/T1w.nii.gz"
#
#                 # eval: load original segmentation (w/o synthesised transformation)
#                 data_path_dict["target_cor_seg"] = f"{self.data_dir}/{tar_subj_id}/cor_seg.nii.gz"
#                 data_path_dict["target_subcor_seg"] = f"{self.data_dir}/{tar_subj_id}/subcor_seg.nii.gz"
#                 data_path_dict["source_cor_seg"] = f"{self.data_dir}/{src_subj_id}/cor_seg.nii.gz"
#                 data_path_dict["source_subcor_seg"] = f"{self.data_dir}/{src_subj_id}/subcor_seg.nii.gz"
#
#         elif self.data_pair == "inter_atlas":
#             subj_id = self.subject_list[index]
#             assert self.atlas_path is not None, "Atlas path not given."
#
#             # images
#             if self.run == "train" and random.choice([True, False]):
#                 # training: randomly choose to use synthesised image as target image
#                 data_path_dict["target"] = f"{self.data_dir}/{subj_id}/T1w_synth.nii.gz"
#             else:
#                 data_path_dict["target"] = f"{self.data_dir}/{subj_id}/T1w.nii.gz"
#
#             data_path_dict["source"] = f"{self.atlas_path}/T2w.nii.gz"  # atlas
#             data_path_dict["roi_mask"] = f"{self.data_dir}/{subj_id}/roi_mask.nii.gz"
#
#             if self.run == "eval":
#                 # T1w image of the other subject for error maps and RMSE
#                 data_path_dict["target_original"] = f"{self.atlas_path}/T1w.nii.gz"
#
#                 # eval: load original segmentation (w/o synthesised transformation)
#                 data_path_dict["target_cor_seg"] = f"{self.data_dir}/{subj_id}/cor_seg.nii.gz"
#                 data_path_dict["target_subcor_seg"] = f"{self.data_dir}/{subj_id}/subcor_seg.nii.gz"
#                 data_path_dict["source_cor_seg"] = f"{self.atlas_path}/cor_seg.nii.gz"  # atlas
#                 data_path_dict["source_subcor_seg"] = f"{self.atlas_path}/subcor_seg.nii.gz"  # atlas
#
#         else:
#             raise ValueError(f"Data pairing setting not recognised: {self.data_pair}")
#
#         return data_path_dict
#
#     def __getitem__(self, index):
#         data_path_dict = self._set_path(index)
#         data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)  # load 2d/3d
#         return self._to_tensor(data_dict)
#

# # Synthesis dataset #
#
# class _SynthDataset(_BaseDataset):
#     def __init__(self,
#                  data_dir,
#                  run,
#                  dim,
#                  slice_range=(70, 90),
#                  sigma=8,
#                  cps=10,
#                  disp_max=1.,
#                  crop_size=(192, 192, 192),
#                  device=torch.device('cpu')  # ??
#                  ):
#         """
#         Loading, scripts and synthesising transformation
#         Args:
#             sigma: (int, float or tuple) sigma of the Gaussian smoothing filter
#             cps: (int, float or tuple) Control point spacing
#             disp_max: (int, float or tuple) Maximum displacement of the control points
#             crop_size: (int or tuple) Size of the image to crop into
#             device: (torch.device)
#         """
#         super(_SynthDataset, self).__init__(data_dir, run, dim, slice_range=slice_range)
#
#         self.crop_size = crop_size  # todo: dimension check to enable integer argument
#         self.device = device
#
#         # elastic parameters
#         self.sigma = sigma
#         self.cps = cps
#         self.disp_max = disp_max
#
#         # Gaussian smoothing filter for random transformation generation
#         self.smooth_filter = GaussianFilter(dim=self.dim, sigma=self.sigma)
#
#     def _set_path(self, index):
#         """ Set the paths of data files to load """
#         raise NotImplementedError
#
#     def _load_2d(self, data_path_dict):
#         """2D axial slices, data shape (N=#slices, H, W)"""
#         data_dict = dict()
#         for name, data_path in data_path_dict.items():
#             data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
#
#         # slice selection
#         if self.run == "train":
#             # randomly select a slice within range
#             z = random.randint(self.slice_range[0], self.slice_range[1])
#             slicer = slice(z, z + 1)  # keep dim
#         else:  # generate
#             # take all slices within range
#             slicer = slice(self.slice_range[0], self.slice_range[1])
#
#         for name, data in data_dict.items():
#             data_dict[name] = data[slicer, ...]  # (N/1, H, W)
#
#         return data_dict
#
#     def _synthesis(self, data_dict):
#         # generate synthesised DVF and deformed T1 image
#         data_dict["target"], data_dict["dvf_gt"] = synthesis_elastic_deformation(data_dict["target_original"],
#                                                                                  data_dict["roi_mask"],
#                                                                                  smooth_filter=self.smooth_filter,
#                                                                                  cps=self.cps,
#                                                                                  disp_max=self.disp_max,
#                                                                                  device=self.device)
#         return data_dict
#
#     def __getitem__(self, index):
#         data_path_dict = self._set_path(index)
#         data_dict = getattr(self, f"_load_{self.dim}d")(data_path_dict)  # load 2d/3d
#         data_dict = self._crop_and_pad(data_dict, self.crop_size)
#         data_dict = self._normalise_intensity(data_dict)
#         data_dict = self._synthesis(data_dict)
#         return self._to_tensor(data_dict)
#
#
# class BratsSynthDataset(_SynthDataset):
#     def __init__(self, *args, **kwargs):
#         super(BratsSynthDataset, self).__init__(*args, **kwargs)
#
#     def _set_path(self, index):
#         subj_id = self.subject_list[index]
#         data_path_dict = dict()
#         data_path_dict["target_original"] = f"{self.data_dir}/{subj_id}/{subj_id}_t1.nii.gz"
#         data_path_dict["source"] = f"{self.data_dir}/{subj_id}/{subj_id}_t2.nii.gz"
#         data_path_dict["roi_mask"] = f"{self.data_dir}/{subj_id}/{subj_id}_brainmask.nii.gz"
#         return data_path_dict
#
#
# class IXISynthDataset(_SynthDataset):
#     def __init__(self, *args, **kwargs):
#         super(IXISynthDataset, self).__init__(*args, **kwargs)
#
#     def _set_path(self, index):
#         subj_id = self.subject_list[index]
#         data_path_dict = dict()
#         data_path_dict["target_original"] = f"{self.data_dir}/{subj_id}/T1-brain.nii.gz"
#         data_path_dict["source"] = f"{self.data_dir}/{subj_id}/T2-brain.nii.gz"
#         data_path_dict["roi_mask"] = f"{self.data_dir}/{subj_id}/T1-brain_mask.nii.gz"
#         return data_path_dict
#
#     @staticmethod
#     def _crop_and_pad(data_dict, crop_size):
#         # todo: this should be deprecated if IXI data is aligned to MNI space
#         # crop by brain mask bounding box for IXI dataset to centre
#         bbox, _ = bbox_from_mask(data_dict["roi_mask"], pad_ratio=0.0)
#         for name, data in data_dict.items():
#             data_dict[name] = bbox_crop(data[:, np.newaxis, ...], bbox)[:, 0, ...]
#
#         # cropping and pad images
#         for name in ["target_original", "source", "roi_mask"]:
#             data_dict[name] = crop_and_pad(data_dict[name], new_size=crop_size)
#         return data_dict
#
#
# class CamCANSynthDataset(_SynthDataset):
#     def __init__(self, *args, **kwargs):
#         super(CamCANSynthDataset, self).__init__(*args, **kwargs)
#
#     def _set_path(self, index):
#         subj_id = self.subject_list[index]
#         data_path_dict = dict()
#         data_path_dict["target_original"] = f"{self.data_dir}/{subj_id}/T1_brain.nii.gz"
#         data_path_dict["source"] = f"{self.data_dir}/{subj_id}/T2_brain.nii.gz"
#         data_path_dict["roi_mask"] = f"{self.data_dir}/{subj_id}/T1_brain_mask.nii.gz"
#
#         # structural segmentation maps
#         data_path_dict["cor_seg"] = f"{self.data_dir}/{subj_id}/fsl_cortical_seg.nii.gz"
#         data_path_dict["subcor_seg"] = f"{self.data_dir}/{subj_id}/fsl_all_fast_firstseg.nii.gz"
#         return data_path_dict


# Legacy function for loading DVF ground truth
# def _load_3d(self, data_path_dict):
#     data_dict = dict()
#     for name, data_path in data_path_dict.items():
#         if name == "dvf_gt":
#             # skip loading ground truth DVF for training
#             if self.run == "train":
#                 continue
#             # dvf_gt is saved in shape (H, W, D, 3) -> (ch=1, 3, H, W, D)
#             data_dict[name] = load_nifti(data_path).transpose(3, 0, 1, 2)[np.newaxis, ...]
#
#         else:
#             # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
#             data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
#     return data_dict

# def _load_2d(self, data_path_dict):
#     data_dict = dict()
#     for name, data_path in data_path_dict.items():
#         if name == "dvf_gt":
#             if self.run == "train":
#                 continue
#             # dvf is saved in shape (H, W, N, 2) -> (N, 2, H, W)
#             data_dict[name] = load_nifti(data_path).transpose(2, 3, 0, 1)
#
#         else:
#             # image is saved in shape (H, W, N) ->  (N, H, W)
#             data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
#
#     # randomly select a slice for training
#     if self.run == "train":
#         z = random.randint(self.slice_range[0], self.slice_range[1])
#         slicer = slice(z, z + 1)  # use slicer to keep dim
#         for name, data in data_dict.items():
#             data_dict[name] = data[slicer, ...]  # (1, H, W)
#     return data_dict

