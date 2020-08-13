"""
Traditional iterative Free-Form Deformation registration algorithm baseline
D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill, M. O. Leach, and D. J. Hawkes. IEEE Transactions on Medical Imaging, 18(8):712-721, 1999.

Implemented using the MIRTK toolkit:
https://mirtk.github.io/
"""
import os
from os import path
import nibabel as nib
import argparse
import numpy as np
import logging
from tqdm import tqdm

import sys
# always run from one dir above ./src
import utils.experiment
import utils.experiment.experiment
import utils.experiment.model

sys.path.insert(0, f'{os.getcwd()}/src')

from archive.runners.helpers import MetricReporter
from utils.metric import calculate_metrics
from utils import misc
from utils.image_io import save_nifti, split_volume_idmat
from utils.transformation import dof_to_dvf

parser = argparse.ArgumentParser()
parser.add_argument('-modality',
                    default='multi',
                    help="mono-modal or multi-modal.")

""" MIRTK FFD parameters """
parser.add_argument('-sim',
                    default='NMI',
                    help="(Dis-)similarity measure of registration.")

parser.add_argument('-CPS',
                    default=8,
                    help="B-spline FFD control point distance at the highest resolution.")

parser.add_argument('-BE',
                    default=1e-4,
                    help="Bending Energy weighting.")

parser.add_argument('-BINS',
                    default=64,
                    help="Number of bins for mutual information.")

parser.add_argument('-verbose',
                    default=0,
                    help="Verbose level of MIRTK FFD.")
""""""

parser.add_argument('--run_dir',
                    default=None,
                    help="Base directory to run FFD.")

parser.add_argument('--data_dir',
                    default='/vol/biomedic2/hq615/PROJECTS/2_mutual_info/data/brats17/miccai2020/test_crop192_sigma8_cps10_dispRange1.5-2.5_sliceRange70-90')

parser.add_argument('--save',
                    action='store_true',
                    help="Save deformed images and predicted DVF etc. if True.")

parser.add_argument('--debug',
                    action='store_true',
                    help="Save intermediate results in True.")

args = parser.parse_args()

# set up FFD model dir
model_dir = misc.setup_dir(f"{args.run_dir}/sim_{args.sim}_CPS_{args.CPS}_BE_{args.BE}")

# save parameters passed to MIRTK via command arguments to a par.conf file
parout = path.join(model_dir, "par.conf")

# set up logger
utils.experiment.set_logger(path.join(model_dir, 'ffd_eval.log'))
logging.info('Starting FFD evaluation...')

# instantiate metric result reporters
metrics_reporter = MetricReporter()
metrics_reporter.id_list = os.listdir(args.data_dir)
metric_groups = ["dvf_metrics", "image_metrics"]  # todo: args or params?

logging.info('Looping over subjects...')
with tqdm(total=len(os.listdir(args.data_dir))) as t:
    for subj_id in sorted(os.listdir(args.data_dir)):
        subj_data_dir = path.join(args.data_dir, subj_id)

        # subject output & temporary working directory
        subj_output_dir = misc.setup_dir(f"{model_dir}/output/{subj_id}")
        subj_tmp_dir = misc.setup_dir(f"{subj_output_dir}/tmp")

        # define target and source images
        # todo: this can be more efficient after standarlising the image names in the generation process
        data_path_dict = dict()
        data_path_dict["target"] = f"{subj_data_dir}/{subj_id}_t1_deformed.nii.gz"
        data_path_dict["target_original"] = f"{subj_data_dir}/{subj_id}_t1.nii.gz"
        data_path_dict["roi_mask"] = f"{subj_data_dir}/{subj_id}_brainmask.nii.gz"
        data_path_dict["dvf_gt"] = f"{subj_data_dir}/{subj_id}_dvf_t2_to_t1_deformed.nii.gz"
        if args.modality == "mono":
            data_path_dict["source"] = data_path_dict["target_original"]
        else:  # multimodal
            data_path_dict["source"] = f"{subj_data_dir}/{subj_id}_t2.nii.gz"

        # create symlinks to original images
        data_link_dict = dict()
        for name, p in data_path_dict.items():
            data_link_dict[name] = f"{subj_output_dir}/{name}.nii.gz"
            os.system(f"ln -sf {p} {data_link_dict[name]}")

        # load nibabel image objects via symlinks
        nim_dict = dict()
        for name, l in data_link_dict.items():
            nim_dict[name] = nib.load(l)

        # load data and transpose to shape (N, H, W) or (N, 2, H, W)
        data_dict = dict()
        for name, nim in nim_dict.items():
            if name == "dvf_gt":
                data_dict[name] = nim.get_data().transpose(2, 3, 0, 1)
            else:
                data_dict[name] = nim.get_data().transpose(2, 0, 1)

        # split volume into 2D slices
        for name in ["target", "target_original", "source"]:
            split_volume_idmat(data_link_dict[name], f'{subj_tmp_dir}/{name}_z')

        # todo: segmentation

        # initialise for output data
        for name in ["dvf_pred", "target_pred", "warped_source"]:
            data_dict[name] = []


        """Looping over slices"""
        for z in range(data_dict["target"].shape[0]):

            # forward registration
            target_z_path = f'{subj_tmp_dir}/target_z{z:02d}.nii.gz'
            source_z_path = f'{subj_tmp_dir}/source_z{z:02d}.nii.gz'
            dof = f'{subj_tmp_dir}/target_to_source_z{z:02d}.dof.gz'
            os.system(f'mirtk register '
                      f'{target_z_path} {source_z_path}  '
                      f'-sim {args.sim} '
                      f'-ds {args.CPS} '
                      f'-be {args.BE} '
                      f'-model FFD '
                      f'-levels 3 '
                      f'-padding -1 '
                      f'-bins {args.BINS} '
                      f'-parout {parout} '
                      f'-dofout {dof} '
                      f'-verbose {args.verbose}')

            # convert the DOF file to dense DDF
            dvf_z = dof_to_dvf(target_z_path, dof, f'dvf_z{z:02d}', subj_tmp_dir)  # (2, H, W)
            data_dict["dvf_pred"].append(dvf_z)

            # MIRTK transform target original image
            target_original_z_path = f'{subj_tmp_dir}/target_original_z{z:02d}.nii.gz'
            target_pred_z_path = f'{subj_tmp_dir}/target_pred_z{z:02d}.nii.gz'
            os.system(f'mirtk transform-image '
                                  f'{target_original_z_path} {target_pred_z_path} '
                                  f'-dofin {dof} '
                                  f'-target {target_z_path}')

            # read warped target original image back in and form numpy arrays
            target_pred_z = nib.load(target_pred_z_path).get_data()[np.newaxis, :, :, 0]  # (1, H, W)
            data_dict["target_pred"].append(target_pred_z)

            # MIRTK transform source image
            warped_source_z_path = f'{subj_tmp_dir}/warped_source_z{z:02d}.nii.gz'
            os.system(f'mirtk transform-image '
                                  f'{source_z_path} {warped_source_z_path} '
                                  f'-dofin {dof} '
                                  f'-target {target_z_path}')

            # read warped source image back in and form numpy arrays
            warped_source_z = nib.load(warped_source_z_path).get_data()[np.newaxis, :, :, 0]  # (1, H, W)
            data_dict["warped_source"].append(warped_source_z)
        """"""

        # slices to stack arrays
        for name in ["dvf_pred", "target_pred", "warped_source"]:
            data_dict[name] = np.array(data_dict[name])  # (N, 1/2, H, W)

        # expand the extra dimension for metric calculation and saving -> (N, 1, H, W)
        for name in ["target", "target_original", "source", "roi_mask"]:
            data_dict[name] = data_dict[name][:, np.newaxis, ...]

        """
        Calculate metrics
        """
        metric_results = calculate_metrics(data_dict, metric_groups)
        metrics_reporter.collect(metric_results)
        """"""

        """ 
        Save predicted DVF and warped images 
        """
        if args.save:
            for name in ["dvf_pred", "target_pred", "warped_source"]:
                save_nifti(data_dict[name].transpose(2, 3, 0, 1),  # (H, W, N, 1/2)
                           f"{subj_output_dir}/{name}.nii.gz",
                           nim_dict["target"])

        # clean up intermediate outputs
        if not args.debug:
            os.system(f"rm -rf {subj_tmp_dir}")

        t.update()


""" Save metrics results """
metrics_reporter.summarise()

# save metric results to JSON files
metrics_reporter.save_mean_std(f"{model_dir}/test_results.json")
metrics_reporter.save_df(f"{model_dir}/test_metrics_results.pkl")

logging.info(f"Evaluation complete. Metric results saved at: \n\t{model_dir}")

