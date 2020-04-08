"""Evaluate FFD on the task of cardiac motion estimation"""
import os
from os import path
import nibabel as nib
import argparse
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd

from data.utils import Normalise
from utils.split_nifti import split_volume_idmat
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack, aee, rmse, rmse_dvf
from utils import misc
from utils.transform import dof_to_dvf
from utils.image import bbox_from_mask

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
model_dir = path.join(args.run_dir, f"sim_{args.sim}_CPS_{args.CPS}_BE_{args.BE}")
if not path.exists(model_dir):
    os.makedirs(model_dir)

# save parameters passed to MIRTK via command arguments to a par.conf file
parout = path.join(model_dir, "par.conf")

# set up logger
misc.set_logger(path.join(model_dir, 'ffd_eval.log'))
logging.info('Starting FFD evaluation...')

# metric result buffers
AEE_buffer = []
RMSE_DVF_buffer = []
RMSE_buffer = []

mean_mag_grad_detJ_buffer = []
negative_detJ_buffer = []
subj_id_buffer = []

# image intensity normaliser
normaliser_minmax = Normalise(mode="minmax")

logging.info('Looping over subjects...')
with tqdm(total=len(os.listdir(args.data_dir))) as t:
    for subj_id in sorted(os.listdir(args.data_dir)):
        subj_data_dir = path.join(args.data_dir, subj_id)

        # subject output directory
        subj_output_dir = path.join(model_dir, "output", subj_id)
        if not path.exists(subj_output_dir):
            os.makedirs(subj_output_dir)

        # subject temporary working directory
        subj_tmp_dir = path.join(subj_output_dir, "tmp")
        if not path.exists(subj_tmp_dir):
            os.makedirs(subj_tmp_dir)

        # save a list of subject id for dataFrames
        subj_id_buffer += [subj_id]

        # define target and source images
        target_data_path = path.join(subj_data_dir, f'{subj_id}_t1_deformed.nii.gz')
        target_original_data_path = path.join(subj_data_dir, f'{subj_id}_t1.nii.gz')
        roi_mask_data_path = path.join(subj_data_dir, f'{subj_id}_brainmask.nii.gz')
        dvf_gt_data_path = path.join(subj_data_dir, f'{subj_id}_dvf_t2_to_t1_deformed.nii.gz')  #(this direction is named wrong)
        if args.modality == "mono":
            source_data_path = target_original_data_path
        else:  # multimodal
            source_data_path = path.join(subj_data_dir, f'{subj_id}_t2.nii.gz')

        # create symlinks to original images
        target_data_link = path.join(subj_output_dir, "target.nii.gz")
        target_original_data_link = path.join(subj_output_dir, "target_original.nii.gz")
        source_data_link = path.join(subj_output_dir, "source.nii.gz")
        roi_mask_data_link = path.join(subj_output_dir, "roi_mask.nii.gz")
        dvf_gt_data_link = path.join(subj_output_dir,  "dvf_gt.nii.gz")

        os.symlink(target_data_path, target_data_link)
        os.symlink(target_original_data_path, target_original_data_link)
        os.symlink(source_data_path, source_data_link)
        os.symlink(roi_mask_data_path, roi_mask_data_link)
        os.symlink(dvf_gt_data_path, dvf_gt_data_link)

        # load in images via symlinks
        nim_target = nib.load(target_data_link)
        nim_target_original = nib.load(target_original_data_link)
        nim_source = nib.load(source_data_link)
        nim_roi_mask = nib.load(roi_mask_data_link)

        # load data, shape (N, H, W)
        target, target_original, source, roi_mask = [x.get_data().transpose(2, 0, 1)
                                                     for x in
                                                     [nim_target, nim_target_original, nim_source, nim_roi_mask]]

        # split volume into 2D slices
        split_volume_idmat(target_data_link, f'{subj_tmp_dir}/target_z')
        split_volume_idmat(target_original_data_link, f'{subj_tmp_dir}/target_original_z')
        split_volume_idmat(source_data_link, f'{subj_tmp_dir}/source_z')

        # todo: segmentation

        dvf_pred_buffer = []
        target_pred_buffer = []
        warped_source_buffer = []

        # loop over the slices
        for z in range(target.shape[0]):

            # todo: modularise inference for each subject for individual use

            # forward registration
            target_img_z_path = f'{subj_tmp_dir}/target_z{z:02d}.nii.gz'
            source_img_z_path = f'{subj_tmp_dir}/source_z{z:02d}.nii.gz'
            dof = f'{subj_tmp_dir}/target_to_source_z{z:02d}.dof.gz'
            os.system(f'mirtk register '
                                  f'{target_img_z_path} {source_img_z_path}  '
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
            dvf_pred_buffer += [dof_to_dvf(target_img_z_path, dof, f'dvf_z{z:02d}', subj_tmp_dir)]

            # MIRTK transform target original image
            target_original_z_path = f'{subj_tmp_dir}/target_original_z{z:02d}.nii.gz'
            target_pred_z_path = f'{subj_tmp_dir}/target_pred_z{z:02d}.nii.gz'
            os.system(f'mirtk transform-image '
                                  f'{target_original_z_path} {target_pred_z_path} '
                                  f'-dofin {dof} '
                                  f'-target {target_img_z_path}')

            # read warped target original image back in and form numpy arrays
            target_pred_buffer += [nib.load(target_pred_z_path).get_data()[:, :, 0]]  # (H, W) each

            # MIRTK transform source image
            warped_source_z_path = f'{subj_tmp_dir}/warped_source_z{z:02d}.nii.gz'
            os.system(f'mirtk transform-image '
                                  f'{source_img_z_path} {warped_source_z_path} '
                                  f'-dofin {dof} '
                                  f'-target {target_img_z_path}')

            # read warped source image back in and form numpy arrays
            warped_source_buffer += [nib.load(warped_source_z_path).get_data()[:, :, 0]]  # (H, W) each

        # slices to volume
        dvf_pred = np.array(dvf_pred_buffer).transpose(0, -1, 1, 2)  # (N, 2, H, W)
        target_pred = np.array(target_pred_buffer)  # (N, H, W)
        warped_source = np.array(warped_source_buffer)  # (N, H, W)

        """Evaluate metrics"""
        # find brian mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask)

        ## DVF accuracy vs. ground truth
        dvf_gt = nib.load(dvf_gt_data_link).get_data().transpose(2, 3, 0, 1)  # (N, 2, H, W)

        # mask both prediction and ground truth DVF with roi mask
        dvf_pred_roi_masked = dvf_pred * roi_mask[:, np.newaxis, ...]  # (N, 2, H, W) * (N, 1, H, W) = (N, 2, H, W)
        dvf_gt_roi_masked = dvf_gt * roi_mask[:, np.newaxis, ...]  # (N, 2, H, W) * (N, 1, H, W) = (N, 2, H, W)

        # crop by roi mask bbox to reduce background
        dvf_pred_roi_bbox_cropped = dvf_pred_roi_masked[:, :,
                                    mask_bbox[0][0]:mask_bbox[0][1],
                                    mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 2, H', W')
        dvf_gt_roi_bbox_cropped = dvf_gt_roi_masked[:, :,
                                  mask_bbox[0][0]:mask_bbox[0][1],
                                  mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 2, H', W')

        AEE = aee(dvf_pred_roi_bbox_cropped, dvf_gt_roi_bbox_cropped)
        print("AEE: ", AEE)
        AEE_buffer += [AEE]
        RMSE_dvf = rmse_dvf(dvf_pred_roi_bbox_cropped, dvf_gt_roi_bbox_cropped)
        print("RMSE(DVF): ", RMSE_dvf)
        RMSE_DVF_buffer += [RMSE_dvf]

        ## RMSE(image)
        target_roi_bbox_cropped = target[:,
                                  mask_bbox[0][0]:mask_bbox[0][1],
                                  mask_bbox[1][0]:mask_bbox[1][1],
                                  ]  # (N, H', W')
        target_pred_roi_bbox_cropped = target_pred[:,
                                       mask_bbox[0][0]:mask_bbox[0][1],
                                       mask_bbox[1][0]:mask_bbox[1][1]]  # (N, H', W')

        RMSE = rmse(target_roi_bbox_cropped, target_pred_roi_bbox_cropped)
        RMSE_buffer += [RMSE]

        # Jacobian related metrics
        mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf_pred.transpose(0, 2, 3, 1),  # (N, H, W, 2)
                                                          rescaleFlow=False)
        mean_mag_grad_detJ_buffer += [mean_grad_detJ]
        negative_detJ_buffer += [mean_negative_detJ]

        """ Save predicted DVF and warped images """
        if args.save:
            nim_dvf = nib.Nifti1Image(dvf_pred.transpose(2, 3, 0, 1),  # (H, W, N, 2) - shape used in NIFTI
                                      nim_target.affine, nim_target.header)
            nib.save(nim_dvf, f"{subj_output_dir}/dvf_pred.nii.gz")

            nim_warped_target_original = nib.Nifti1Image(target_pred.transpose(1, 2, 0),
                                                         nim_target.affine, nim_target.header)
            nib.save(nim_warped_target_original, f"{subj_output_dir}/target_pred.nii.gz")

            nim_warped_source = nib.Nifti1Image(warped_source.transpose(1, 2, 0),
                                                         nim_target.affine, nim_target.header)
            nib.save(nim_warped_source, f"{subj_output_dir}/warped_source.nii.gz")

        # clean up intermediate outputs
        if not args.debug:
            os.system(f"rm -rf {subj_tmp_dir}")

        t.update()


""" Save metrics results """
## JSON file for mean and std
# construct metrics dict
results_dict = {}

# RMSE (dvf) and RMSE (image)
rmse_criteria = ["AEE", "RMSE_DVF", "RMSE"]
for cr in rmse_criteria:
    result_name = cr
    the_buffer = locals()[f'{result_name}_buffer']
    results_dict[f'{result_name}_mean'] = np.mean(the_buffer)
    results_dict[f'{result_name}_std'] = np.std(the_buffer)

# regularity
reg_criteria = ['mean_mag_grad_detJ', 'negative_detJ']
for cr in reg_criteria:
    result_name = cr
    the_buffer = locals()[f'{result_name}_buffer']
    results_dict[f'{result_name}_mean'] = np.mean(the_buffer)
    results_dict[f'{result_name}_std'] = np.std(the_buffer)

# sanity check: proportion of negative Jacobian points should be lower than 1
assert results_dict['negative_detJ_mean'] <= 1, "Invalid det Jac: Ratio of folding points > 1"

# save
save_path = path.join(model_dir, "test_results.json")
misc.save_dict_to_json(results_dict, save_path)


## data frame with result for individual test subjects (for boxplots)
df_buffer = []
column_method = ['FFD'] * len(subj_id_buffer)

# save RMSE and AEE
rmse_data = {'Method': column_method,
             'ID': subj_id_buffer,
             'RMSE': RMSE_buffer,
             'RMSE_DVF': RMSE_DVF_buffer,
             'AEE': AEE_buffer}
rmse_df = pd.DataFrame(data=rmse_data)
rmse_df.to_pickle(f"{model_dir}/FFD_test_results_rmse.pkl")

# save detJac metrics
jac_data = {'Method': column_method,
            'ID': subj_id_buffer,
            'GradDetJac': mean_mag_grad_detJ_buffer,
            'NegDetJac': negative_detJ_buffer}
jac_df = pd.DataFrame(data=jac_data)
jac_df.to_pickle(f"{model_dir}/FFD_test_results_Jacobian.pkl")

logging.info("Evaluation complete. Metric results saved at: \n\t{}".format(model_dir))
