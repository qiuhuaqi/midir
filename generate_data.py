"""Generate synthetic deformation & save data for validation and evaluation"""
import os
import os.path as path
from tqdm import tqdm
import argparse
import numpy as np

from data.datasets import BratsDataset
from utils.image_io import save_nifti

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default="/vol/biomedic2/hq615/PROJECTS/2_mutual_info/data/brats17/3d")

parser.add_argument("-dim",
                    default=None,
                    type=int,
                    help="Data dimension, 2/3")

parser.add_argument("-cps",
                    nargs='*',
                    default=(10, 10, 10),
                    type=int,
                    help="Control point spacing of the synthesis model.")

parser.add_argument("-sigma",
                    nargs='*',
                    default=(1, 1, 1),
                    type=float,
                    help="Sigma for Gaussian filter in deformation synthesis")

parser.add_argument("-disp_max",
                    nargs='*',
                    type=float,
                    default=[4., 4., 4.],
                    help="Maximum displacements in each direction in voxel space")

parser.add_argument("-crop_size",
                    nargs='*',
                    type=int,
                    default=[192, 192, 160],
                    help="Central crop size")

parser.add_argument("-slice_range",
                    nargs=2,
                    type=int,
                    default=[70, 90],
                    help="Range of slice numbers (axial plane), 2D only")

parser.add_argument("--debug", action="store_true", help="Debug mode.")

args = parser.parse_args()





# set random seed
# (this seed means the deformation generated for validation set is the same as test set,
# but the image data is different so this should be fine)
# todo: fix random seeding in dataset
np.random.seed(12)

for run in ["val", "test"]:
    print(f"Generating: {run} dataset...")

    data_original_dir = f"{args.data_dir}/{run}_original"
    output_dir = f"{args.data_dir}/{run}"
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # construct the dataset
    brats_dataset = BratsDataset(data_original_dir,
                                 run=run,
                                 dim=args.dim,
                                 sigma=args.sigma,
                                 cps=args.cps,
                                 disp_max=args.disp_max,
                                 crop_size=args.crop_size,
                                 slice_range=args.slice_range
                                 )

    if args.debug:
        print(data_original_dir)
        print(len(brats_dataset))
        print(brats_dataset.subject_list)


    with tqdm(total=len(brats_dataset)) as t:
        for idx, data_dict in enumerate(brats_dataset):
            """
            Note:
            - images are minmax normalised to [0, 1]
            - dvf is not normalised (in number of pixels)
            - all cropped to crop_size
            """

            subj_id = brats_dataset.subject_list[idx]
            output_subj_dir = output_dir + f"/{subj_id}"
            if not path.exists(output_subj_dir):
                os.makedirs(output_subj_dir)

            for name, data in data_dict.items():
                if name == "dvf_gt":
                    if args.dim == 2:  # 2D
                        # (N, 2, H, W) -> (H, W, N, 2)
                        _data = data.numpy().transpose(2, 3, 0, 1)  # (H, W, N, 2)
                    else:  # 3D
                        # (1, 3, H, W, D) -> (H, W, D, 3)
                        _data = data[0, ...].numpy().transpose(1, 2, 3, 0)

                else:
                    if args.dim == 2:  # 2D
                        # (N, H, W) -> (H, W, N)
                        _data = data.numpy().transpose(1, 2, 0)
                    else:  # 3D
                        # (1, H, W, D) -> (H, W, D)
                        _data = data.numpy()[0, ...]

                # save image and DVF
                save_nifti(_data, f"{output_subj_dir}/{name}.nii.gz")


            # debug: check intensity ranges
            if args.debug:
                print("Subject ", subj_id)
                for name, data in data_dict.items():
                    print(f"{name}, Shape: {data.shape}, "
                          f"Intensity range: [{data.min()}, {data.max()}], "
                          f"Mean: {data.mean()}, Std: {data.std()}")
            t.update()
