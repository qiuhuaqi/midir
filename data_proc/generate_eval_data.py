"""Generate synthetic deformation & save data for validation and evaluation"""
import os
import os.path as path

import sys

sys.path.insert(1, "./src")

from data.datasets import Brats2D

import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/vol/biomedic2/hq615/PROJECTS/1_cardiac_motion/data/brats17/miccai2020")

parser.add_argument("-cps", default=10, type=int, help="Control point spacing of the synthesis model.")
parser.add_argument("-sigma", default=8, type=float, help="Sigma for Gaussian filter in deformation synthesis")
parser.add_argument("-disp_range", nargs=2, type=float, default=[1.5, 2.5],
                    help="Range of displacement in number of pixels for control points, 'min, max'")
parser.add_argument("-slice_range", nargs=2, type=int, default=[70, 90],
                    help="Range of slice numbers (axial direction) used.")
parser.add_argument("-crop_size", nargs='?', type=int, default=192,
                    help="Central crop size. If only one argument is given, all dimensions are cropped to the same size")
args = parser.parse_args()

# set random seed
# (this seed means the deformation generated for validation set is the same as test set,
# but the image data is different so this should be fine)
np.random.seed(8)

data_root_dir = args.data_dir

# parameters (elastic deformation)
sigma = args.sigma
cps = args.cps
disp_range = args.disp_range
slice_range = args.slice_range
crop_size = args.crop_size


for val_test in ["val", "test"]:
    print(f"Generating: {val_test} dataset...")

    data_dir = f"/vol/biomedic2/hq615/PROJECTS/1_cardiac_motion/data/brats17/miccai2020/{val_test}"
    output_dir = data_root_dir + \
                 f"/{val_test}_crop{crop_size}_sigma{sigma}_cps{cps}_dispRange{disp_range[0]}-{disp_range[1]}_sliceRange{slice_range[0]}-{slice_range[1]}"
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # construct the dataset
    brats2d_val = Brats2D(data_dir,
                          run="generate",
                          slice_range=slice_range,
                          sigma=sigma,
                          cps=cps,
                          disp_range=disp_range,
                          crop_size=crop_size)

    with tqdm(total=len(brats2d_val)) as t:
        for idx, data_point in enumerate(brats2d_val):
            target, source, target_original, brain_mask, dvf = data_point

            # to numpy array
            target = target.numpy().transpose(1, 2, 0)  # (HxWxN)
            source = source.numpy().transpose(1, 2, 0)  # (HxWxN)
            target_original = target_original.numpy().transpose(1, 2, 0)  # (HxWxN)
            brain_mask = brain_mask.numpy().transpose(1, 2, 0)  # (HxWxN)
            dvf = dvf.numpy().transpose(2, 3, 0, 1)  # (HxWxNx2)
            """
            Note:
            - images are not normalised
            - dvf is not normalised (in number of pixels), only normalise DVF before spatial transformation
            - all cropped
            """

            # save
            subj_id = brats2d_val.subject_list[idx]
            output_subj_dir = output_dir + f"/{subj_id}"
            if not path.exists(output_subj_dir):
                os.makedirs(output_subj_dir)

            nim_target = nib.Nifti1Image(target, np.eye(4))
            nib.save(nim_target, output_subj_dir + f"/{subj_id}_t1_deformed.nii.gz")

            nim_source = nib.Nifti1Image(source, np.eye(4))
            nib.save(nim_source, output_subj_dir + f"/{subj_id}_t2.nii.gz")

            nim_target_original = nib.Nifti1Image(target_original, np.eye(4))
            nib.save(nim_target_original, output_subj_dir + f"/{subj_id}_t1.nii.gz")

            nim_brain_mask = nib.Nifti1Image(brain_mask, np.eye(4))
            nib.save(nim_brain_mask, output_subj_dir + f"/{subj_id}_brainmask.nii.gz")

            nim_dvf = nib.Nifti1Image(dvf, np.eye(4))
            nib.save(nim_dvf, output_subj_dir + f"/{subj_id}_dvf_t2_to_t1_deformed.nii.gz")

            t.update()
